---
name: build-flowsheet
description: Build, change or check an aquakin plant flowsheet (the YAML file of units, connections and settings a project's studies run on). Use when the user asks to add, remove or rewire units, change unit settings, or set up a new treatment plant in aquakin.
---

# Building or changing an aquakin flowsheet

Use the aquakin MCP tools; never write the flowsheet YAML by hand, because
`write_flowsheet` checks every unit and fills in the catalogue defaults.

1. **Read before you write.** If the project has a flowsheet, `read_flowsheet`
   it and `check_flowsheet` it, so you know its units, wiring and any
   problems before you change anything.
2. **Find the units.** `list_unit_kinds`, then `describe_unit_kind` for each
   kind you use: it gives every setting (type, default, help), the ports and
   which settings are exclusive. Models come from `list_models` and
   `describe_model`. Do not guess setting names.
3. **Check each new unit** with `check_unit` (kind and settings): it reports
   the unit's real ports, or why it cannot be built.
4. **Wire it.** Connections are `"<unit>.<port> -> <unit>.<port>"`; every
   input must be fed. For the conventions (recycles, clarifiers, splitters,
   influents) read the guide: `read_doc_page` on `custom_flowsheets`, and
   `influents` for feeds.
5. **Check the whole plant** with `check_flowsheet(spec=...)` before writing.
   Fix every problem it reports.
6. **Write it** with `write_flowsheet`. Writing over an existing file needs
   `replace=true`: only pass it when the user asked to change that file.
   Tell the user what changed; if they have the editor open, it shows the
   new version by itself.

Keep the user's unit names and layout: change only what was asked.
