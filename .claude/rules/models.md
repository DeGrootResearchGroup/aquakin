---
paths:
  - "aquakin/models/**"
---

# Rules — `aquakin/models/`

When adding or editing a shipped model:

- **`docs/model_catalog.md`** — the full per-model catalog (what each shipped
  model is, its provenance, and the documented corrections). Read it before
  touching a model so you do not reintroduce a fixed import bug.
- **`.claude/rules/schema.md`** — the YAML schema rules (also auto-loads for any
  `*.yaml`): stoichiometry expressions, `extends:`, `auto`/`?` coefficient
  resolution, `composition:`, `speciation:`, `precipitation:`, positivity.
- Run `scripts/verify_sumo_asm.py` after any hand-edit to a SUMO-derived ASM model.
- Per the comment convention (see root `CLAUDE.md`), scientific provenance goes in
  a model YAML's `references:` block, not in code comments.
