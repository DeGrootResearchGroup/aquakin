---
name: run-study
description: Set up, run and read an aquakin study (a saved simulation, Fisher information, calibration, profile likelihood or sensor ranking of a flowsheet or a batch reactor). Use when the user asks to simulate a plant, fit parameters, check identifiability, rank sensors, or rerun or compare existing studies.
---

# Running an aquakin study

A study is a folder under `<project>/studies/<name>/` with its settings,
results and run record. Use the aquakin MCP tools for every step.

1. **See what exists.** `list_studies` on the project: each study's kind,
   state (fresh, stale and why, not run, failed) and whether it is running.
   Reuse or copy (`copy_study`) a study rather than making a near duplicate.
2. **Know the settings.** `study_settings` lists every kind and its fields,
   with which are required; `study_choices` lists the streams, parameters
   (`"<model>.<parameter>"`) and measured channels a study of the flowsheet
   may name. For what the analyses mean, read the guide: `read_doc_page` on
   `plant_files` (studies), `calibration`, `plant_calibration`,
   `experiment_design` or `analysis_concepts`.
3. **Create or edit** with `create_study` / `update_study`. They validate the
   settings exactly as a run reads them, so fix what they report.
4. **Run** with `run_study`. It returns once the run has started; it goes on
   in its own process. Poll `study_status` (its `running.progress`) until
   `running` is null, then read `state` and `reasons`.
   - A study with results is not run again unless `replace=true`; only pass
     it when the user asked, and offer `copy_study` to keep the old results.
   - A study runs once at a time; `cancel_study` stops a run.
5. **Read** with `study_results` and report the numbers the user asked for,
   with their units, and any notes (stale inputs, a newer aquakin).

Never unfreeze a study with results (`discard_results=true`) or replace
results without the user's explicit go-ahead: both delete their work.
