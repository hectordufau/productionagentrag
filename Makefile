PYTHON ?= python3

setup:
	$(PYTHON) -m pip install -e '.[dev]'

up:
	docker compose up --build

down:
	docker compose down

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check .

format:
	ruff format .

benchmark:
	$(PYTHON) scripts/run_benchmark.py
