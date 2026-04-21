PYTHON ?= python3

.PHONY: lint test build smoke

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest

build:
	$(PYTHON) -m build

smoke:
	PYTHONPATH=src $(PYTHON) -m lean_swarm.cli smoke

