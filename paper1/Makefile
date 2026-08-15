# ═══════════════════════════════════════════════════════════
# Makefile for LLM Post-Training Efficiency Paper
# Template: ICML 2026 (from Orchestra-Research/AI-Research-SKILLs)
# ═══════════════════════════════════════════════════════════

PAPER = main
LATEX = pdflatex
BIBTEX = bibtex

# Source files
SECTIONS = $(wildcard sections/*.tex)
TABLES = $(wildcard tables/*.tex)
FIGURES = $(wildcard figures/*.pdf)

.PHONY: all figures clean distclean

# Default target: build paper with figures
all: figures $(PAPER).pdf

# Build PDF (3 passes for references)
$(PAPER).pdf: $(PAPER).tex $(SECTIONS) $(TABLES) $(FIGURES) refs.bib
	$(LATEX) -interaction=nonstopmode $(PAPER)
	$(BIBTEX) $(PAPER) || true
	$(LATEX) -interaction=nonstopmode $(PAPER)
	$(LATEX) -interaction=nonstopmode $(PAPER)
	@echo "✓ Built $(PAPER).pdf"

# Generate figures from JSON data
figures:
	cd figures && python3 generate_all.py

# Quick build (no figures regeneration)
quick: $(PAPER).tex $(SECTIONS) $(TABLES) refs.bib
	$(LATEX) -interaction=nonstopmode $(PAPER)
	$(BIBTEX) $(PAPER) || true
	$(LATEX) -interaction=nonstopmode $(PAPER)
	$(LATEX) -interaction=nonstopmode $(PAPER)

# Clean auxiliary files
clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.synctex.gz
	rm -f sections/*.aux tables/*.aux

# Full clean (including PDF)
distclean: clean
	rm -f $(PAPER).pdf
	rm -f figures/*.pdf

# View PDF (Linux)
view: $(PAPER).pdf
	xdg-open $(PAPER).pdf &

# Count pages
pages: $(PAPER).pdf
	@pdfinfo $(PAPER).pdf | grep Pages || echo "pdfinfo not available"

# Check for overfull warnings
check: $(PAPER).log
	@grep -i "overfull" $(PAPER).log | head -20 || echo "No overfull warnings"
	@grep -i "underfull" $(PAPER).log | head -10 || echo "No underfull warnings"
