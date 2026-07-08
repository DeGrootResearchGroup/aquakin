# Minimal Sphinx makefile for local doc builds.
#
#   make html      -> build the HTML site into _build/html
#   make doctest   -> run the doctest blocks in the docs
#   make linkcheck -> check external links
#
# CI builds the strict-warnings site directly with
#   sphinx-build -b html -W docs docs/_build/html

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html doctest linkcheck clean Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)" generated api.md

# Catch-all: route any other target (html, doctest, linkcheck, ...) to
# sphinx-build's -M mode.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
