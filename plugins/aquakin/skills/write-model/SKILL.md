---
name: write-model
description: Write or change an aquakin kinetic model (a YAML file of species, parameters, reactions, rate expressions and stoichiometry), or extend a shipped model. Use when the user asks for a new reaction model, a change to rates or stoichiometry, or a variant of ASM, ADM or another shipped model.
---

# Writing an aquakin model

1. **Start from what exists.** `list_models` and `describe_model`: if a
   shipped model is close, extend it (`extends:`) rather than copying it.
   `read_model_file` shows a shipped model's YAML as a worked example.
2. **Follow the format exactly.** Read `read_doc_page` on `model_format`
   (every key, the rate-expression grammar, built-in functions, stoichiometry,
   `composition:` and `auto` coefficients, temperature corrections, `extends:`)
   and `adding_models` (the walkthrough and common mistakes). Do not invent
   keys or functions.
3. **Keep rate constants as parameters**, with units, bounds and a reference
   value; never write a number into a rate expression that a user might fit.
4. **Check it**: `describe_model(path=...)` loads it and shows the reactions
   and stoichiometry; read them back against what was intended. In Claude
   Code, also run Python: `aquakin.load_model_from_file(path)`, then
   `model.check_conservation()` and `model.check_units()`, and fix what they
   report.
5. Put the literature source in the model's `references:` block.
