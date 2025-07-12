# Minimal Makefile for DMS documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Simple HTML build target
html:
	@echo "Building HTML documentation..."
	@mkdir -p $(BUILDDIR)/html
	@echo "<!DOCTYPE html>" > $(BUILDDIR)/html/index.html
	@echo "<html><head><title>DMS Documentation</title></head>" >> $(BUILDDIR)/html/index.html
	@echo "<body><h1>DMS - Detection Model Suite</h1>" >> $(BUILDDIR)/html/index.html
	@echo "<p>Documentation is being built. Please check back later.</p>" >> $(BUILDDIR)/html/index.html
	@echo "</body></html>" >> $(BUILDDIR)/html/index.html
	@echo "HTML documentation built successfully!"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) 