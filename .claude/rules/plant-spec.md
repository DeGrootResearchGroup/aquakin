---
paths:
  - "aquakin/plant/spec.py"
  - "aquakin/plant/catalog.py"
  - "aquakin/plant/time_expr.py"
  - "aquakin/plant/influent.py"
  - "aquakin/__main__.py"
---

# Rules — flowsheet spec, unit catalogue, time expressions and influents

Split out of `.claude/rules/plant.md`.

**Expressions of time (`plant/time_expr.py`).** An operating value that varies on
a schedule — an influent's flow, a setpoint — may be written as a function of the
plant clock (`"18446 + 4000*sin(2*pi*t)"`). `compile_time_expression(value,
time_scale=, what=)` reads one into a callable; a plain number compiles to a
constant, so a value that does not vary costs nothing. It compiles to plain JAX
arithmetic with **no Python control flow**, so it traces, differentiates and
batches inside the RHS; there is deliberately **no comparison or conditional**
(a value that switches uses `clip` or `step`), because an `if` on a traced value
cannot run. `time_scale` is the factor from the caller's clock to the clock the
expression was written on. **The grammar is shared in spirit with
`core/parser.py` and follows Python's precedence**, including that a sign binds
*looser* than a power (`-2**2` is `-(2**2)`): `unary := ('+'|'-') unary | power`
and `power := primary ('**' unary)?`. Both parsers were changed together in
#699 — they had bound a sign tighter, which nobody predicts; keep them in step.
`sqrt` and `log` are **not** bound raw: they come from `core/kernels.py`, the
guarded definitions the rate language uses, because a clock starts every solve
at `t = 0` and that is exactly where their derivative is infinite. An unguarded
`sqrt(t)` leaves the forward solve correct and breaks only the gradient, which
surfaces as an exhausted step budget pointing at the integrator (#700 / the
shared-lexer PR); `log10` is written as `safe_log(x)/ln(10)` so there is one
guarded logarithm, not two. New functions go in `FUNCTIONS`
(name → `(callable, arities)`), which the parser, the arity check and the
error messages all read, so they cannot drift. A function that is associative is
named in `core/precedence.ASSOCIATIVE_CALLS` rather than given a second arity: a
call carrying more arguments than it declares then folds left, so `max(a, b, c)`
is `max(max(a, b), c)` in **both** languages. The whole Python-agreement
contract — what matches and what deliberately does not — is pinned by
`tests/unit/test_python_consistency.py`; add to its corpus rather than writing a
one-off test when you touch the grammar.

**Influents: a record or a pattern.** `InfluentSeries` is sampled data
(`at(t)` interpolates); **`InfluentExpression`** is the same feed written as
expressions of time, for a load that follows a shape rather than a measurement.
Both satisfy the only seam the plant uses — `at(t)`, `.model`,
`in_time_unit(tu)` — so nothing in `plant.py` / `recycle.py` branches on which it
is. Differences to keep in mind: a declared unit is applied **as the expression
is evaluated** (folded into the compiled callable) rather than to stored arrays,
since a string cannot be scaled in place; and `in_time_unit` converts the `t` the
expressions are *given* and the flow they *return*, never the expression, so one
written in days still means days on an hourly plant. The catalogue's `Q`, `T` and
`composition` are `scheduled_number` / `scheduled_species_values`: `_influent`
builds an `InfluentExpression` when **any** value is an expression and otherwise
the sampled constant series exactly as before, so every existing spec reads back
bit-identically. `spec._influent_entry` writes a pattern straight back as its own
settings (no `data/` table).

**Unit catalogue (`plant/catalog.py`, the `aquakin` command).** Front ends
(the flowsheet editor prototype) learn the unit types from
`aquakin.plant.catalog`, never by importing unit classes. A kind's *settings* are
read from the class: constructor signature (types mapped to front-end types
such as `number`, `port_name`, `aeration`, `records`, `tagged`, `unsupported`), dataclass default
factories, and the docstring's `Parameters` section for help. Only what the
signature cannot say is declared per kind in `_build_kinds`: label, category,
port rules beyond the naming convention (`input_port`/`input_port_names` are
inputs, every other `*_port` and the port-keyed mappings are outputs, in
constructor order; fixed ports such as the splitters' `in`, the mixer's `out`,
the storage tank's `out`/`bypass` are `PortRule(form="fixed")`), GUI-friendly
defaults for required port settings, and `exactly_one_of` groups.
**A setting whose value is a class's constructor arguments** reads and writes them through one set of functions: `record_settings` / `record_plain` (fill every argument from its default, check each by name and type, refuse an unknown one) and `kind_settings` / `kind_plain` for a mapping that picks the class by `kind`. A connection's translator, an SBR's `phases` (`records`, the classes in `_RECORDS`) and its `settling` (`tagged`, in `_TAGGED`) all use them, and the catalogue sends each such setting's `fields` (per kind for `tagged`); a new one is a row in those tables, not another reader. `filled_value` is the canonical form (every argument written), so leaving a default out does not change `spec_fingerprint`. `check_unit` builds the real unit and is the authority. **When adding a unit
class or changing a unit's ports**, add or update its kind:
`tests/unit/test_plant_catalog.py` builds every kind and compares the
rule-derived ports with the built unit's, and fails if an exported unit class
has no kind. `aquakin serve` (`aquakin/__main__.py`) is a JSON-lines worker
(`catalog`, `models`, `model`, `check_unit`, `check_spec`, `fingerprint`,
`read_spec`, `write_spec`, `ping`).
Replies go to a private duplicate of the process's stdout and descriptor 1 is
pointed at stderr while serving (plus a Python-level redirect per handler), so
no other output can corrupt a reply; a failed request, or a reply that is not
strict JSON (NaN), gets `{"ok": false, "error": ...}` without stopping the worker.
**An error is `{"error": message, "type": class name}`**
(`plant.errors.error_reply`): a refused request, `check_unit` / `check_spec`
results and the study run's error event all use it. Never format
`f"{type(exc).__name__}: {exc}"` into a message a front end shows. The commands
`aquakin_ui` sends, and the study states and kinds it names, are pinned by
`tests/ui/test_protocol.py` (see `.claude/rules/ui.md`).

**Flowsheet spec (`plant/spec.py`, `Plant.to_spec` / `Plant.from_spec`).** A
plant as a file (format 2), built on the catalogue. Units are keyed by their
stable name (the plant's unit name; `Plant.labels` holds optional display text)
and written with **every** setting (`catalog.unit_settings(full=True)`; unit
attributes must mirror the constructor arguments for this to work) plus the
aquakin version, so a later change of default cannot change a saved plant.
Connections are `"unit.port -> unit.port"` or `{from, to, translator: {kind,
...}, seed: {Q, T, concentrations}}`; a seed is written in the **destination**
unit's model (it stands for the stream arriving after any translator). A `kind:
influent` unit is an `InfluentSeries` wired with `add_influent`: constant
(`Q`/`T`/`composition`) or a `series` table file (`read_influent_table`, which
records the file on the series; `write_influent_table` round-trips bit for bit)
recorded under the spec's `files:` with its sha256. `to_spec(path)` refers to a
table inside the spec's directory and otherwise writes `data/<name>.csv`; without
a path a time-varying influent is refused. **`write_spec`** (behind `to_spec(path)`
and the editor's Save) plans everything before touching the disk: each unit's
settings are checked with `validate_settings` (unset required settings are allowed,
so a draft saves) and canonicalized (`_canonical_settings`: every number a float,
infinities `"inf"`), the output is checked as strict JSON (no NaN), and the planned
copies are checked for conflicts (`_check_writes`); only then are files copied and
the spec replaced atomically. A model file whose `extends:` chain leaves the
project folder has its **whole chain** copied into `models/` (`_plan_model`: the
model as `<name>.yaml`, each base beside it with its `extends:` line rewritten
textually so comments survive; a project model is rewritten in place). An
in-memory influent table is rendered to bytes first, so saving twice is
idempotent. `spec_fingerprint(spec)` hashes what is simulated (canonical settings
with catalogue defaults filled in, wiring, streams, endpoints, model and table
content fingerprints) and leaves out layout, labels, name, file locations and
the aquakin version; results should record it to detect staleness. NaN is no
setting anywhere: `catalog._is_number` and the Pydantic schema
(`allow_inf_nan=False`) refuse it and `_plain(nan)` raises. A file model may not
take a shipped model's name (refused on read and write), and
`composition_base` gives a same-named model the shipped table only when the
species match. Parse errors from `load_spec_file` are `SpecError`s. Every
model is listed under `models:` as `shipped` or a YAML file path relative to the
spec, with a sha256 of its content (`extends` resolved; a mismatch on read is a
`SpecWarning`). A model is recognized by the provenance `load_model` /
`load_model_from_file` record on it (`_aquakin_source`, a plain attribute, so a
`dataclasses.replace` copy such as `bsm2_asm1_model()` does not carry it and is
refused rather than written under the shipped name); this survives
`clear_model_cache`. Reading validates the document with Pydantic and each
unit's settings strictly with `catalog.validate_settings` (unknown keys with a
did-you-mean, exact types), and reads YAML with YAML 1.2 scalars
(`_Yaml12Loader`). Controllers derived from a closed-loop `Aeration` / feedback
`DosingUnit` are recorded in `plant._derived_units` by `_materialize_aeration` /
`_materialize_dosing` and left out (with their measurement taps), since they are
derived again. Every refusal is a `SpecError` naming the unit / connection.
`check_spec(spec, base_dir=)` builds the plant and returns `Plant.check` (the
`aquakin serve` `check_spec` command). `Plant.add_unit` / `add_influent` refuse
names with `.`, `->` or whitespace and a unit named after one of the plant's
models (`_check_unit_name`; such a name made `parameter_index("asm1.muH")` return
a flow setpoint), and `_plant_param_index` raises on any duplicate key. Pinned by
`tests/integration/test_plant_spec.py`: BSM1 (ideal and Takács) and a two-model
plant (translators, temperature seed, reagent, influent T) rebuild with an
identical spec, initial state and bit-identical RHS, and the
`examples/projects/bsm2` project (its 15 °C ASM1 a model file extending `asm1`)
rebuilds `build_bsm2`'s plant run with the BSM2 parameters bit-identically.
