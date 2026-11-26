---
paths:
  - "README.md"
  - "docs/index.md"
  - "docs/getting_started.md"
  - "docs/models.md"
  - "docs/simulation.md"
  - "docs/analysis.md"
  - "docs/reference.md"
  - "docs/reactors.md"
  - "docs/plants.md"
  - "docs/sensitivity_and_calibration.md"
  - "docs/model_format.md"
  - "docs/adding_models.md"
  - "docs/model_catalog.md"
  - "docs/public_api.md"
  - "conftest.py"
---

# Rules — the code examples in the README and the user guide

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
- **A page of whole-plant solves is `slow`**: add it to `SLOW_PAGES` in the root
  `conftest.py`. A new user-guide page goes in `GUIDE_PAGES` there and in this
  file's `paths:`.
- Run a page locally with `pytest docs/<page>.md` (or `pytest README.md`). Keep
  the pages out of `testpaths` and out of the pytest-split shards: a page's
  blocks share a namespace, so they must run on one worker (`--dist loadfile`
  under xdist).
