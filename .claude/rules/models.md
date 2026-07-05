---
paths:
  - "aquakin/models/**"
---

# Rules — `aquakin/models/`

When adding or editing a shipped model:

- **`docs/model_catalog.md`** — the full per-model catalog (what each shipped
  model is, its provenance, and the documented corrections). Read it before
  touching a model so you do not reintroduce a fixed import bug.
- **`docs/khalil_reproduction_log.md`** — the JRN-055 Khalil model-improvement
  sequence (the chronological structural-correction log for the WATS/Khalil family).
- **`.claude/rules/schema.md`** — the YAML schema rules (also auto-loads for any
  `*.yaml`): stoichiometry expressions, `extends:`, `auto`/`?` coefficient
  resolution, `composition:`, `speciation:`, `precipitation:`, positivity.
- The `_make_*.py` generators in this directory are run manually and are **not**
  imported by the package or tests (so `ruamel` is intentionally not a runtime dep).
  Re-run the relevant generator after editing a base it splices from, and run
  `scripts/verify_sumo_asm.py` after any hand-edit to a SUMO-derived ASM model.
- Per the comment convention (see root `CLAUDE.md`), scientific provenance goes in
  a model YAML's `references:` block, not in code comments.
