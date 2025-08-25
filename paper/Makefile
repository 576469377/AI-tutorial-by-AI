# Makefile for AI Tutorial by AI Academic Paper
# Author: AI Tutorial Development Team
# Date: 2024

# Variables
MAIN = main
TEX_FILES = $(wildcard sections/*.tex)
BIB_FILE = bibliography.bib
PDF_OUTPUT = paper.pdf
FIGURES_DIR = figures
TABLES_DIR = tables
SCRIPTS_DIR = scripts

# Default target
all: pdf

# Generate figures and tables
data: figures tables

# Generate figures
figures:
	@echo "Generating figures..."
	@mkdir -p $(FIGURES_DIR)
	cd $(SCRIPTS_DIR) && python generate_figures.py

# Generate tables
tables:
	@echo "Generating tables..."
	@mkdir -p $(TABLES_DIR)
	cd $(SCRIPTS_DIR) && python generate_tables.py

# Compile PDF
pdf: $(PDF_OUTPUT)

$(PDF_OUTPUT): $(MAIN).tex $(TEX_FILES) $(BIB_FILE)
	@echo "Compiling LaTeX document..."
	pdflatex $(MAIN).tex
	bibtex $(MAIN)
	pdflatex $(MAIN).tex
	pdflatex $(MAIN).tex
	@if [ -f $(MAIN).pdf ]; then mv $(MAIN).pdf $(PDF_OUTPUT); fi
	@echo "PDF generated: $(PDF_OUTPUT)"

# Quick compile (no bibliography)
quick:
	@echo "Quick compile (no bibliography)..."
	pdflatex $(MAIN).tex
	@if [ -f $(MAIN).pdf ]; then mv $(MAIN).pdf $(PDF_OUTPUT); fi

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

# Clean everything including PDF
clean-all: clean
	@echo "Cleaning all generated files..."
	rm -f $(PDF_OUTPUT)
	rm -rf $(FIGURES_DIR)/*.png $(FIGURES_DIR)/*.pdf $(FIGURES_DIR)/*.eps
	rm -rf $(TABLES_DIR)/*.tex $(TABLES_DIR)/*.csv

# Watch for changes and recompile
watch:
	@echo "Watching for changes..."
	@which inotifywait > /dev/null || (echo "inotify-tools not found. Install with: sudo apt-get install inotify-tools" && exit 1)
	@while true; do \
		inotifywait -e modify $(MAIN).tex $(TEX_FILES) $(BIB_FILE); \
		echo "File changed, recompiling..."; \
		make pdf; \
	done

# Install dependencies (Ubuntu/Debian)
install-deps:
	@echo "Installing LaTeX dependencies..."
	sudo apt-get update
	sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra
	sudo apt-get install -y texlive-latex-recommended texlive-science
	pip install -r ../requirements.txt

# Check if LaTeX is installed
check-latex:
	@which pdflatex > /dev/null || (echo "pdflatex not found. Install LaTeX first." && exit 1)
	@which bibtex > /dev/null || (echo "bibtex not found. Install LaTeX first." && exit 1)
	@echo "LaTeX installation verified."

# Development build (with data generation)
dev: data pdf

# Full build from scratch
rebuild: clean-all dev

# Spell check (requires aspell)
spell:
	@echo "Checking spelling..."
	@which aspell > /dev/null || (echo "aspell not found. Install with: sudo apt-get install aspell" && exit 1)
	aspell check $(MAIN).tex
	@for file in $(TEX_FILES); do \
		echo "Checking $$file..."; \
		aspell check $$file; \
	done

# Count words
wordcount:
	@echo "Word count:"
	@texcount -brief $(MAIN).tex

# Show paper statistics
stats:
	@echo "Paper Statistics:"
	@echo "================"
	@echo "Main file: $(MAIN).tex"
	@echo "Section files: $(shell echo $(TEX_FILES) | wc -w)"
	@echo "Bibliography entries: $(shell grep -c '^@' $(BIB_FILE) 2>/dev/null || echo 0)"
	@if [ -f $(PDF_OUTPUT) ]; then \
		echo "PDF size: $(shell ls -lh $(PDF_OUTPUT) | awk '{print $$5}')"; \
	fi
	@echo "Generated figures: $(shell ls $(FIGURES_DIR)/*.png 2>/dev/null | wc -l)"
	@echo "Generated tables: $(shell ls $(TABLES_DIR)/*.tex 2>/dev/null | wc -l)"

# Help
help:
	@echo "AI Tutorial by AI - Academic Paper Build System"
	@echo "=============================================="
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Generate data and compile PDF (default)"
	@echo "  pdf          - Compile PDF from LaTeX sources"
	@echo "  quick        - Quick compile without bibliography"
	@echo "  data         - Generate figures and tables"
	@echo "  figures      - Generate figures only"
	@echo "  tables       - Generate tables only"
	@echo "  dev          - Development build (data + pdf)"
	@echo "  rebuild      - Clean and rebuild everything"
	@echo "  clean        - Remove auxiliary files"
	@echo "  clean-all    - Remove all generated files"
	@echo "  watch        - Watch files and auto-recompile"
	@echo "  spell        - Check spelling"
	@echo "  wordcount    - Count words in document"
	@echo "  stats        - Show paper statistics"
	@echo "  check-latex  - Verify LaTeX installation"
	@echo "  install-deps - Install dependencies (Ubuntu/Debian)"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make pdf               # Basic compilation"
	@echo "  make dev               # Full development build"
	@echo "  make watch             # Auto-recompile on changes"
	@echo "  make clean && make pdf # Clean compile"

.PHONY: all pdf quick data figures tables clean clean-all watch install-deps check-latex dev rebuild spell wordcount stats help