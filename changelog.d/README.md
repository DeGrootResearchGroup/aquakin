# Changelog fragments

Each user-visible change adds **one file here** instead of editing
`CHANGELOG.md`. Because every change writes its own file, two pull requests never
conflict over the changelog. At release, `towncrier build --version X.Y.Z`
collects the files into a new `CHANGELOG.md` section and deletes them (see
`RELEASING.md`).

## Naming

`<name>.<type>.md`, where `<type>` is the Keep a Changelog group:

| type | for |
|---|---|
| `added` | new features, models, public API |
| `changed` | changes to existing behaviour |
| `deprecated` | features that will be removed |
| `removed` | features removed |
| `fixed` | bug fixes |
| `security` | vulnerabilities |

`<name>` is the GitHub issue the change resolves (`742.fixed.md` renders with a
link to issue #742), or, when there is no issue, `+` and a short slug
(`+calibrate-scales-the-mse-objective.fixed.md`). Two changes for the same issue
and type use `742.fixed.md` and `742.fixed.1.md`.

## Content

Write the entry as one Markdown list item without its leading `- `: the text a
user deciding whether to upgrade needs, in Canadian English. Wrap lines at about
80 columns and do not indent continuation lines; towncrier adds the bullet and
the indentation. Entries are sorted by file name within each type.

```text
changelog.d/742.fixed.md:

`direct_n2o_emission` works on a plant whose model names dissolved oxygen
something other than `SO`, such as `asm3_2step_n2o`.
```

Changes with no user-visible effect (internal refactors, tests, CI, docs) need
no fragment.

Preview the next release's section with
`towncrier build --draft --version X.Y.Z` (from the `release` extra:
`pip install -e ".[release]"`).
