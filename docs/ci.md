# Continuous integration

The CI architecture: the fast PR gate vs the merge-to-main suite, pytest-split
sharding, the memory/heavy-runner rationale, the `full-ci` / `skip-heavy`
labels, and branch-protection. Reference; the actionable summary is in the
Development workflow section of the root `CLAUDE.md`.


GitHub Actions (`.github/workflows/ci.yml`) splits the suite into a **fast PR
gate** and a **full merge-to-main suite**, because the integration tests are
real stiff-ODE solves with AD (~irreducible seconds each) and the cost is spread
across hundreds of tests — neither a JAX compile-cache (verified: cold ≈ warm)
nor concentrating it in a few files makes the whole suite fast.

- The **`test`** job (`pytest -m "not validation and not slow"`, Python
  3.11/3.12) runs on **every PR and every push** — unit + fast integration. It is
  the merge gate, and is **`pytest-split`-sharded** (`--splits 4 --group i`, `-n
  auto` within each shard) for the *same* reason the slow/validation suites are:
  the fast suite grew to ~1300 tests and a single long-lived run accumulated XLA
  cache + live JAX buffers until an xdist worker was OOM-killed (a worker *crash*
  on a random lightweight test) and/or it hit the time ceiling — intermittent red
  on the required gate with no real regression. Four fresh shards × 2 Python
  versions bound each shard's footprint to ~1/4 (no OOM) and run in parallel (well
  under the ceiling). The sharded per-version jobs are **not** the required check —
  the **`fast-gate`** aggregator (job name **`fast gate`**) is: it `needs:` the
  shard matrix and is green iff every shard succeeded (or was skipped — a bare
  `labeled` event skips the gate, which must not block). It is a *stable* required-
  check name that survives shard-count changes, so **branch protection requires
  `fast gate`**, not the per-shard `fast tests (...)` jobs.
- The **`slow`** job (`pytest -m "slow and not validation and not heavy"`,
  3.11/3.12) and **`validation`** job (`pytest -m "validation and not heavy"`,
  3.12) run **only on push to `main`** (`if: github.event_name == 'push'`). They carry the multi-minute
  stiff/plant solves (the `slow` marker on `test_bsm2_dynamic`, `test_bsm1`,
  `test_biofilm`, `test_forward_sensitivity`, the two `test_wats_sewer_*` files)
  and the published-data checks. A regression a PR's fast gate cannot catch
  therefore surfaces within minutes of merging — revert from there. **Both are
  `pytest-split`-sharded** (`--splits N --group i`, `-n 1` within each shard):
  even serially, a single long-lived process accumulating the whole set's XLA
  compilation cache + live JAX buffers exhausts the 16 GB hosted runner and is
  OOM-reclaimed mid-suite (~62 min in, SIGTERM / exit 143, before the timeout) —
  first the validation set, then the slow set once enough whole-plant tests
  landed. Sharding across N fresh processes bounds each process's footprint to
  ~1/N (slow: 8 shards × 2 Python versions; validation: 4 shards, 3.12). The
  partition is complete and disjoint, so coverage is unchanged. (pytest-split
  balances by `.test_durations` where recorded — now including the heaviest slow
  tests — else evenly by count; duration-balancing evens the wall time.) **But
  sharding alone is loose memory control: more shards or better balancing only
  *relocate* an overweight shard** (a new whole-plant-AD test once walked the OOM
  4/6 → 5/8 across re-shard / re-balance attempts), because the accumulation is
  per-shard, not per-test. A per-test cache-clear fixture (`jax.clear_caches()` +
  `gc.collect()` after every `slow` test, optionally with a `malloc_trim`
  follow-up) was tried and **rejected** — do not re-add it: profiling showed
  `clear_caches()` itself *spikes* RSS and only partially reclaims (the freed
  compiled programs are not returned to the OS, so the process RSS still creeps up
  test-by-test), and `malloc_trim` had nothing to reclaim because the programs stay
  live until GC. The accumulation is intrinsic to the **whole-plant
  stable-adjoint gradient tests**: each compiles a multi-GB plant program, and a
  shard's worth piles up faster than any between-test clear can reclaim,
  OOM-killing the runner regardless of shard count (a single such test runs fine
  alone — the failure is cumulative). So those tests carry the **`heavy`** marker
  and are excluded from every slow / validation / smoke / durations job (`... and
  not heavy`). They run instead on the dedicated **`heavy` job**, sharded
  **one-test-per-process** (`--splits 7 -n 1`) across the free 16 GB
  `ubuntu-latest` runner: a single heavy test fits 16 GB (it peaks near the limit
  but completes), and one-per-process is exactly what prevents the cumulative
  accumulation, so **no paid larger runner is needed**. This replaced a paid
  16-core/64 GB `aquakin-heavy` larger runner whose per-minute billing was the
  entire CI cost — public-repo standard-runner minutes are free. The heavy set is
  the 7 BSM2 plant-gradient checks; the lighter BSM1 plant-AD gradient tests are
  **not** heavy (they carry only `slow` / `validation` and run in those free jobs).
  *(If a single heavy test ever exceeds 16 GB, the fallback is an 8-core/32 GB
  larger runner at one-per-process — still about half the per-minute cost of the
  old 16-core/64 GB tier — not more sharding, which cannot shrink a single test's
  peak.)*
- The **`heavy`** job (`pytest -m heavy -n 1`, 3.12) runs on `ubuntu-latest`,
  sharded 7 ways (`--splits 7 --group i`), with two triggers: **push to `main`**
  when the **`heavy-gate`** path filter sees a change under `aquakin/plant`,
  `aquakin/integrate`, `aquakin/core`, `aquakin/schema`, `aquakin/networks`, the
  two heavy test files, `pyproject.toml` or the workflow (so a docs / examples /
  utils / unrelated-test merge does not run it at all); and a **`full-ci` PR** (the
  pre-merge opt-in, so the adjoint machinery is validated before merge). The
  **`skip-heavy`** label still force-skips a push-to-main run for a change that
  touches a filtered path but cannot affect the adjoint. `heavy-gate` is
  **fail-safe**: when the path filter cannot determine a base it treats every path
  as changed (heavy runs), and a run is skipped only when `skip-heavy` is
  positively present.
- The **`smoke`** job (`pytest -m "slow and not validation and not heavy" --splits
  18 --group <rotating>`, 3.12) runs on **every PR** as an early-warning slice of the
  merge-only `slow` set: it *executes* a bounded ~1/18 shard (~8 tests) so
  shared-fixture breaks, whole-plant call-site regressions and memory creep show
  up before merge, not after. It is deliberately probabilistic — a single PR
  runs only ~8 of the ~141 slow tests — but the shard **rotates by run number**,
  so consecutive runs cover the whole set, and a fixture break (which hits most
  slow tests) is caught by any shard. Scope is `slow and not validation`: with no
  recorded durations those tests split evenly by count (no empty shard, unlike a
  duration-skewed slow+validation split) and the heaviest ~4-min published-data
  tests are excluded, keeping the slice time predictable. It complements the
  fast-gate signature-contract test (`tests/unit/test_api_signatures.py`), which
  catches the *signature* sub-case deterministically; the smoke adds breadth.
- **The `full-ci` label** runs the full merge suite (the `slow` **and**
  `validation` jobs) on a PR *before* merge. Apply it to a PR (the workflow
  listens for the `labeled` event, so no fresh push is needed) and the slow
  whole-plant solves and published-data validation run before it lands. Without
  the label those jobs run only **after** merge to `main`, so a regression they
  catch surfaces post-merge and is reverted from there; the label moves that
  signal earlier at the cost of the runtime. The bare `labeled` event
  deliberately does **not** re-run the fast gate / smoke (they already ran on the
  latest commit) and does not cancel an in-progress run, so labelling never
  disturbs the required checks.
  - **Convention — apply `full-ci` to any PR that touches convergence-sensitive
    code:** the integrators / adjoints, the PTC steady-state solver
    (`plant/steady.py`), the plant assembly / recycle resolution, a network's
    stoichiometry, the pH / precipitation solvers, or the metric / mass-balance
    kernels. Those are exactly the changes whose regressions live in the `slow` /
    `validation` suites, which **do not run on the PR fast gate** — so the
    fast-gate-plus-rotating-smoke coverage can pass while a whole-plant solve or a
    published-data check is broken. (Concretely: a brittle slow test added in
    #394 — a BSM2 PTC cold-start convergence with a platform-sensitive iteration
    count — passed the fast gate and broke `main` only on the post-merge slow run;
    `full-ci` before merge would have caught it. The fix was to make the test
    deterministic, but the label is the process guard.) Skipping the label is fine
    only for changes that cannot reach the slow/validation paths (docs, a new
    isolated unit, a fast-gated network add).

**Branch protection:** the required status check must be the **`fast gate`**
aggregator job — **not** the per-shard `fast tests (py3.x shard i/4)` jobs (their
names/count change when the shard count is tuned) and **not** `slow`/`validation`
(which do not run on PRs and would otherwise block every PR forever). The
`fast gate` job is green only when every fast shard passes, so requiring it gives
the same gate with a stable name. (When the fast gate was sharded, the old
required checks `fast tests (py3.11)` / `(py3.12)` ceased to exist; branch
protection must be repointed to `fast gate` or merges block forever waiting on
checks that never report.)

A green fast gate is the merge gate. Python 3.10 stays
install-compatible (`requires-python >= 3.10`) but is **not** CI-tested:
its heavier jaxlib 0.6.2 build ran close to the hosted runner's
resource/time limits and the job was intermittently killed mid-run, while
3.11/3.12 stayed green and the suite passed locally on 3.10. When adding a runtime
dependency, add it to `pyproject.toml` `dependencies` so the CI install
(`pip install -e ".[test]"`) picks it up — the `_make_*` network
generators' `ruamel` need is intentionally *not* a runtime dep (they are
run manually, never imported by the package or tests). `pandas` is an
**optional** dependency (the `dataframe` extra, for the `to_dataframe()` /
`to_csv()` result exporters), but it is also in the `test` extra so the fast
gate exercises those exporters; the library imports it lazily inside the
exporters (`require_pandas`), so solving never needs it.

---

