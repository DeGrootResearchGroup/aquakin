---
name: analyze-with-python
description: Write and run Python that uses the aquakin library for analyses no study covers, such as batch reactor simulations, reactor calibration, sensitivity analysis, Fisher information and experiment design, uncertainty propagation, or plots. Use when the user asks for an aquakin analysis in Python or a script they can rerun.
---

# Analyses in Python with aquakin

1. **Look up, do not guess.** Find the task in the guide (`search_docs`,
   then `read_doc_page`: `reactors`, `sensitivity_and_calibration`,
   `calibration`, `fit_uncertainty`, `experiment_design`,
   `worked_example_design`), and the exact signature of every function you
   call with `describe_api` (for example `"calibrate"`, `"Plant.calibrate"`,
   `"optimize_experiment"`). `public_api` maps tasks to functions.
2. **Write a script the user can rerun**: imports at the top, inputs as named
   variables, results printed with their units and figures saved with
   `savefig` (no `plt.show()`). Save it in the project when the user may run
   it again.
3. **Import `aquakin` before `jax`**: aquakin turns on 64-bit floats, which
   stiff solves need.
4. **Run it** (in Claude Code, with Python; in other apps, with the aquakin
   `run_python` tool when the server was started with `--allow-python`), read
   the output, and fix errors from the message rather than by guessing.
5. When a saved, reproducible analysis fits a study kind (simulate, fisher,
   calibrate, profile, rank), prefer a study: see the run-study skill.
