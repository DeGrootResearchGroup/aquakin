---
paths:
  - "tests/**"
---

# Rules — `tests/`

How tests are written here. The aim is a suite that **fails when the code is
wrong**; a test that passes for any plausible wrong answer is worth nothing and
hides the gap from the coverage number.

## A test must be able to fail

- **Assert the value, against something independent of the code under test**: a
  closed form, a conservation law, a published number, a finite difference, a
  second independent implementation, or a hand-built expected structure. In that
  order of preference.
- **`isfinite`, `is not None`, `isinstance`, a shape, or "did not raise" is never
  the only assertion** of a test of numerical behaviour. They are fine as extra
  lines beside a value check, and as the whole test only when the property under
  test *is* the type, the shape or the absence of a warning (say so in the name).
- **The name and docstring state what is asserted.** A test called
  `test_..._mass_balance` asserts the balance closes to a tolerance; if it only
  checks that the solve ran, call it a smoke test or, better, add the assertion.
- **Before trusting a new test, break the code and watch it fail.** For a bug
  fix, run the regression test against the unfixed code first (stash the fix, or
  revert the one line) and confirm it fails for the right reason. For a new
  feature, perturb the implementation (flip a sign, change a constant) once and
  confirm a test notices. Say in the hand-off that this was done, or that it
  could not be.
- **Tolerances are justified by the method, not tuned until green.** A closed
  form or a bit-identical refactor gets `rtol <= 1e-8`; an adaptive solve gets a
  tolerance a decade or two above its `rtol`; a finite difference gets what its
  step supports. Anything looser than `1e-2` needs a comment giving the reason.
  Never loosen a tolerance, widen a bound, or add `abs=` to make a failing test
  pass: find out why it moved. If the new value is right, say why in the commit.
- **Do not weaken, skip, `xfail` or delete a failing test to get green.** An
  `xfail` is only for a documented open defect, is `strict=True`, and names the
  defect. A `skip` needs a condition that is actually about the environment.
- **Errors are tested by message**: `pytest.raises(SomeError, match="...")` with
  the specific exception class and a fragment of the message that identifies the
  branch. Never `pytest.raises(Exception)`, and not a bare
  `pytest.raises(ValueError)` where the function has several `ValueError`s.
  Every new `raise` gets a test.

## Gradients

Every integration suite that solves has an AD test, and it **checks the value**:

- against the closed form where there is one (first-order decay:
  `d[A]/dk = -t [A]0 exp(-k t)`), otherwise
- against a central finite difference of the same loss (state the step and use a
  tolerance it supports), or against a second AD route (forward vs reverse, the
  stable adjoint vs the JAX adjoint).

"Finite and the right sign" is the minimum for a whole-plant solve too expensive
to difference in the fast gate, and then the same code path needs a value check
on a small analogue (the single-CSTR plant in `conftest.py`). Many older suites
assert finiteness only; when you touch one, upgrade its gradient test rather
than copying it.

## Reuse before writing

- **Look in `tests/conftest.py` first.** `simple_model`, `make_cstr_plant` /
  `cstr_plant` and `FIXTURES` are there. Use `FIXTURES / "x.yaml"`, never the
  cwd-relative `"tests/fixtures/x.yaml"`.
- **A fixture or helper needed by a second file moves to `conftest.py`** (or a
  `tests/` helper module) in the same change; it is not copied. Known debt to
  pay down as files are touched, not to add to: per-file `asm1` / `adm1` model
  fixtures, local `_single_cstr_plant` / `_constant_influent` builders that
  duplicate `make_cstr_plant`, and the identical `reactor` and stream helpers
  repeated across the Fisher / observation and BSM2 files.
- **The same checks over several models or cases are one parametrized test**
  (`pytest.mark.parametrize`, as `test_asm_family.py` does), not a file per
  model with the tests pasted in.
- Tests may use private names to *inspect* (`plant._jit_cache`), but a test that
  needs a private hook to *set up* is telling you the public seam is missing.
- Do not depend on execution order or leave process-global state changed
  (`clear_model_cache()`, x64, warnings filters); restore it in a fixture.

## Tiers and what actually runs

| Tier | Selected by | Runs |
|---|---|---|
| fast | `tests/unit`, `tests/integration` and `tests/ui`, `-m "not validation and not slow"` | every PR (the merge gate) |
| slow | `@pytest.mark.slow`, or use of a fixture named in `_SLOW_PLANT_FIXTURES` (`conftest.py`) | merge to `main`, and `full-ci` PRs |
| validation | `@pytest.mark.validation` | merge to `main`, and `full-ci` PRs |
| heavy / xheavy | `@pytest.mark.heavy` (`xheavy` too for the one test over 16 GB) | path-filtered, one test per process |

- **A test in `tests/validation/` without `@pytest.mark.validation` runs in no CI
  job**: the fast gate does not look in that directory and the validation job
  selects by marker. Mark every test there (a module-level
  `pytestmark = pytest.mark.validation` is simplest).
- `tests/ui/` runs in the fast gate as its own unsharded `ui` job, the only job
  that imports the editor and measures `aquakin_ui` coverage.
- Keep the fast gate meaningful: a feature whose only tests are `slow` is
  unprotected on a normal PR. Give it a small-plant or unit-level test in the
  fast tier that exercises the same code path, and keep the full-scale one slow.
- New test file in `tests/integration` or `tests/unit`: the shard split reads
  `.test_durations`; CI refreshes it, so do not edit it by hand.

## What to test for a change

| Change | Test |
|---|---|
| bug fix | a regression test that fails on the old code |
| new public function or option | a value test of the behaviour, each error branch by message, an AD test if it is differentiated |
| new AST node / rate function | `tests/unit/test_nodes.py` (value, gradient at the edge of the domain), parser round trip, kernel-vs-scalar equality in `test_vector_kernel.py` |
| new plant unit | mass balance across the unit, steady state against a hand calculation, spec round trip, catalogue entry |
| new shipped model | `check_conservation` clean, a validation test against the published data |
| refactor with no behaviour change | no new test, but the existing ones must pass **unchanged**; editing a test in a refactor needs a stated reason |

Run what you changed before handing off (`pytest path/to/test_file.py`, then the
fast gate for anything cross-cutting) and report the actual outcome. If a test
could not be run here, say so plainly rather than implying it passed.
