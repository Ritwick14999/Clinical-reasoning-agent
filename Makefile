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
#   make data       download PubMedQA + MIRAGE source data
#   make index      build the BM25 retrieval corpus
#   make rollout CONFIG=headline_qwen3   run an experiment, requires Ollama
#   make eval       score committed traces           [Phase 3]
#   make results    regenerate every table and figure from committed traces [Phase 3]

PY := .venv/bin/python
PIP := uv pip

.PHONY: setup test lint fmt demo doctor data index rollout clean

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

data:
	$(PY) -m cra.data.download

index:
	$(PY) -m cra.retrieval.index_build

rollout:
	$(PY) -m cra.cli rollout --config $(CONFIG)

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
