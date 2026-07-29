.PHONY: install test demo paper clean

install:
	python -m pip install -e .[dev]

test:
	pytest

demo:
	./scripts/reproduce_demo.sh

paper:
	python scripts/export_latex.py
	cd paper && pdflatex main.tex && pdflatex main.tex

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
