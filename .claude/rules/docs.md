---
paths:
  - "README.md"
  - "docs/conf.py"
  - "docs/index.md"
  - "docs/getting_started.md"
  - "docs/models.md"
  - "docs/simulation.md"
  - "docs/analysis.md"
  - "docs/reference.md"
  - "docs/reactors.md"
  - "docs/plants.md"
  - "docs/custom_flowsheets.md"
  - "docs/influents.md"
  - "docs/plant_files.md"
  - "docs/installing_the_editor.md"
  - "docs/ai_assistants.md"
  - "docs/analysis_concepts.md"
  - "docs/sensitivity_and_calibration.md"
  - "docs/calibration.md"
  - "docs/fit_uncertainty.md"
  - "docs/experiment_design.md"
  - "docs/plant_calibration.md"
  - "docs/worked_example_design.md"
  - "docs/model_format.md"
  - "docs/adding_models.md"
  - "docs/model_catalog.md"
  - "docs/public_api.md"
  - "conftest.py"
---

# Rules — the user documentation

The published site is the Sphinx build of `docs/` (pages not in `conf.py`'s
`exclude_patterns`). `tests/unit/test_docs_structure.py` checks the mechanical
parts of these rules; the rest is judgement, so read this before adding or
moving content.

## Where content goes

- **Five sections, no more.** `index.md`'s hidden root toctree lists exactly the
  section landing pages (`getting_started`, `models`, `simulation`, `analysis`,
  `reference`); they are the top bar and the phone menu. A new page goes in one
  section's toctree, never in `index.md`'s. When every page was a root entry
  the theme showed them as one flat list and the sidebar was empty.
- **A landing page is a map, not content**: one short paragraph, an "If you
  want to… | Read" table with a row linking every page of its toctree, and the
  toctree. A new page gets its row in the same change.
- **One topic, one place.** Before writing about a feature, `grep docs/` for
  it. Extend the page that owns the task and link to it from elsewhere; do not
  explain it again on a second page (integrator settings live only in
  `reactors.md`, measured channels only in `plant_calibration.md`). Two copies
  drift apart.
- **Guide pages are by task, reference pages are for lookup.** A walkthrough
  goes in a guide page; the exhaustive list of keys, functions or classes goes
  in `model_format.md`, `public_api.md` or a docstring (which `api.md`
  renders), and the guide links to it.
- **Describe the current behaviour.** No history ("now", "previously", "since
  0.2", "was renamed"): that is the changelog's. Keep dev-internal detail out
  (`CLAUDE.md`, "Published documentation site").

## Page size and splitting

- **A page stays within `PAGE_BUDGET` lines** (the test; `model_format.md` is a
  recorded exception and may not grow). Past it, split the page by task into
  sibling pages of the same section; do not shorten it by deleting content a
  reader needs, or move only its tail onto an "advanced" page.
- **When you split a page**: the first part keeps the file name; each new page
  opens with its own setup block so it runs on its own (below); every link to an
  anchor that moved is updated (`sphinx-build -W` reports a missing one); the
  new pages go in `GUIDE_PAGES`, `SLOW_PAGES` if they take minutes, this
  file's `paths:`, and their landing page's table.
- **Do not rename or delete a published page.** Its file name is its URL, which
  the README, PyPI and other people's links point at.

## The code examples are tests

Every fenced `python` block in `README.md` and the user-guide pages is a test:
Sybil (configured in the root `conftest.py`) runs a page's blocks in order in
one namespace, and CI runs them in the `docs-examples` job (the merge gate) and,
for the slow pages, `docs-examples-slow`. An example that stops working fails
CI instead of reaching a reader.

- **A page runs top to bottom as written.** A block may use only what the page
  defined above it. Do not rely on another page's variables, and do not import
  `jax` before `aquakin` (it triggers the x64 warning every reader would see).
- **A value in a comment is checked, or it is not stated.** When a comment says
  what a line prints (`# effluent SNH: 0.46`), add a hidden check after the
  block: in the README an HTML comment
  `<!-- invisible-code-block: python ... -->`, in a MyST page
  `% invisible-code-block: python` followed by `%     ` -indented lines. Round
  to the precision the comment shows.
- **Skip only what cannot run in CI, and say why**: `% skip: next "reason"`
  (MyST) or `<!-- skip: next "reason" -->` (README). A whole-plant BSM2 solve or
  a calibration needing data the page does not supply qualifies; "it failed"
  does not — fix the example or the code.
- **A file the example loads is written by the page.** A `yaml` block whose
  first line is `# <name>.yaml` is written to that file before the following
  python blocks run; any other `yaml` block is only parsed. Pages run in a
  temporary directory, so writing files is fine.
- **Every published page is in `GUIDE_PAGES`** in the root `conftest.py` and in
  this file's `paths:` (the test checks both). A page whose examples take
  minutes is also in `SLOW_PAGES`; keep a page out of it when it runs in
  seconds, so the PR gate tests it.
- Run a page locally with `pytest docs/<page>.md` (or `pytest README.md`). Keep
  the pages out of `testpaths` and out of the pytest-split shards: a page's
  blocks share a namespace, so they must run on one worker (`--dist loadfile`
  under xdist).
