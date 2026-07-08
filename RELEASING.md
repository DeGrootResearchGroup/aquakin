# Releasing aquakin

How a maintainer cuts a release. Publishing to PyPI is automated via **PyPI
Trusted Publishing (OIDC)** — no API tokens are stored; a tagged GitHub Release
triggers the build + upload, with PEP 740 provenance attestations. The pipeline
lives in [`.github/workflows/release.yml`](.github/workflows/release.yml).

The version is single-sourced from `aquakin.__version__`
(`aquakin/__init__.py`); the release tag must be `v<that version>` or the build
job fails the guard.

---

## One-time setup (already done — recorded for the record)

- **PyPI + TestPyPI Trusted Publisher** registered for this repo: owner
  `DeGrootResearchGroup`, repository `aquakin`, workflow `release.yml`,
  environment `pypi` (on PyPI) / `testpypi` (on TestPyPI).
- **GitHub Environments** `pypi` (protected: required reviewer + deployment
  restricted to `v*` tags) and `testpypi` (unprotected — rehearsal only). No
  environment secrets: Trusted Publishing is token-less by design.
- **Read the Docs** project connected; builds on every push to `main` and on
  tags.

---

## Release checklist

Run from a clean, green `main`.

### 1. Pre-flight

- [ ] `main` is green (fast gate, lint, docs). Cut releases from `main`, not a
      branch.
- [ ] Decide the version `X.Y.Z` (SemVer). For `0.1.0` this is the first release.

### 2. Stamp the version, changelog, and citation (one PR)

On a release-prep branch:

- [ ] **Version** — set `__version__ = "X.Y.Z"` in `aquakin/__init__.py` (already
      `0.1.0` for the first release; bump for later ones).
- [ ] **CHANGELOG.md** — rename the running `## [Unreleased]` heading to
      `## [X.Y.Z] - YYYY-MM-DD` (today's date, ISO). Open a fresh empty
      `## [Unreleased]` section above it, and update the link references at the
      bottom of the file:

      ```markdown
      ## [Unreleased]

      ## [X.Y.Z] - YYYY-MM-DD
      ... (the curated notes) ...

      [Unreleased]: https://github.com/DeGrootResearchGroup/aquakin/compare/vX.Y.Z...HEAD
      [X.Y.Z]: https://github.com/DeGrootResearchGroup/aquakin/releases/tag/vX.Y.Z
      ```

      (For `0.1.0`, the notes are the curated initial-release feature summary;
      granular per-change logging under `[Unreleased]` begins for the *next*
      release.)
- [ ] **CITATION.cff** — add `version: X.Y.Z` and `date-released: "YYYY-MM-DD"`
      (the file has a comment marking where).
- [ ] Open the PR, get it green, merge to `main`.

### 3. Rehearse on TestPyPI (recommended)

- [ ] **Actions → Release → Run workflow** (a manual `workflow_dispatch`) —
      builds and uploads to **TestPyPI**.
- [ ] In a clean venv, confirm it installs:

      ```bash
      python -m venv /tmp/aqk && /tmp/aqk/bin/pip install \
        -i https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple/ aquakin
      /tmp/aqk/bin/python -c "import aquakin; print(aquakin.__version__)"
      ```

      (The extra index lets the real deps — jax, diffrax, … — resolve from PyPI
      while `aquakin` comes from TestPyPI.)

      TestPyPI keeps every uploaded version, so the rehearsal is a one-time proof
      the OIDC path works; you do not have to re-run it every release.

### 4. Tag and release → publishes to PyPI

- [ ] Tag the merged commit and push the tag:

      ```bash
      git checkout main && git pull
      git tag -a vX.Y.Z -m "aquakin X.Y.Z"
      git push origin vX.Y.Z
      ```
- [ ] Cut a **GitHub Release** from `vX.Y.Z` (Releases → Draft a new release →
      choose the tag). Paste the `X.Y.Z` CHANGELOG section as the notes and
      publish it.
- [ ] Publishing the Release triggers `release.yml`: it builds, checks the tag
      matches `__version__`, re-verifies the bundled model data is in both the
      sdist and the wheel, then the **`publish-pypi`** job waits on the `pypi`
      environment's required reviewer. **Approve the deployment** (Actions →
      the run → Review deployments).

### 5. Post-release verification

- [ ] Fresh-venv install from real PyPI:

      ```bash
      python -m venv /tmp/aqk2 && /tmp/aqk2/bin/pip install aquakin
      /tmp/aqk2/bin/python -c "import aquakin; print(aquakin.__version__)"
      ```
- [ ] Read the Docs shows the tagged version and the build is green
      (<https://aquakin.readthedocs.io>).
- [ ] The PyPI page shows the version, the README as the description, and the
      provenance attestations.

### 6. Zenodo DOI (post-release, once minted)

The Zenodo webhook mints a DOI when the GitHub Release is published. Once it
exists:

- [ ] Add it to `CITATION.cff` (`doi: 10.5281/zenodo.XXXXXXX`) and add a DOI
      badge to `README.md`. This is a small follow-up PR after the release.

---

## If something goes wrong

- **Wrong tag / version mismatch** — the build job fails the `v<version>` guard
  before anything is uploaded. Delete the tag (`git push --delete origin
  vX.Y.Z`), fix `__version__`, re-tag.
- **A bad artifact reached PyPI** — a PyPI version is **immutable and cannot be
  re-uploaded**. You can *yank* it (hides it from new installs without breaking
  pins) and ship a fixed `X.Y.(Z+1)`. Rehearse fixes on TestPyPI first.
