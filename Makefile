# Reproducibility entry points (POSIX). On Windows use the equivalent
# cross-platform runner instead -- `make` and .venv/bin/python do not exist there:
#
#   python tasks.py setup | test | lint | demo | doctor
#
#   make setup      install into .venv (Python 3.11)
#   make test       unit + end-to-end tests (no network, no credentials)
#   make demo       one agent episode on the mock client
#
# Stages added in later phases:
#   make data       download + prepare datasets and build the retrieval index
#   make rollouts   run the agent (requires model credentials)
#   make eval       score committed traces
#   make results    regenerate every table and figure from committed traces

PY := .venv/bin/python
PIP := uv pip

.PHONY: setup test lint fmt demo doctor clean

setup:
	uv venv --python 3.11 .venv
	$(PIP) install -e ".[dev]"
	@echo "Optional extras: $(PIP) install -e '.[api]'   # API rollouts"
	@echo "                 $(PIP) install -e '.[dense]' # dense retrieval + NLI"
	@echo "                 $(PIP) install -e '.[demo]'  # Gradio demo"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests scripts

fmt:
	$(PY) -m ruff check --fix src tests scripts

demo:
	$(PY) scripts/demo_episode.py

doctor:
	$(PY) tasks.py doctor

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
