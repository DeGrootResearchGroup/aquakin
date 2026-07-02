# Adding a built-in model

Built-in models ship as YAML files under `aquakin/models/`. Adding one:

1. Author the YAML file. See [`model_format.md`](model_format.md) for the
   schema and [`aquakin/models/ozone_bromate.yaml`](../aquakin/models/ozone_bromate.yaml)
   for a worked example.
2. Verify it loads:
   ```python
   import aquakin
   net = aquakin.load_model("my_model")
   print(net.summary())
   ```
3. Add a unit-test fixture covering the new model's structural assertions
   (species count, stoichiometry of key reactions).
4. If experimental trajectories are available, add a validation test under
   `tests/validation/` decorated with `@pytest.mark.validation`.
5. Update the README to list the new model.

## Authoring tips

- Lead with `description:` and `references:` blocks. Future contributors will
  thank you.
- Every reaction should carry a `reference:` line citing the rate-constant
  source.
- For acid/base speciation, use `pH_switch(pKa)` rather than baking pH into
  rate constants. Keeps the model usable across pH ranges.
- For temperature-dependent rates, use `arrhenius(A, Ea)` so that the model
  works at any T declared as a condition.
- Use `bounds:` on parameters when ranges are known from the literature.
  `aquakin.fit` will respect them as box constraints.
