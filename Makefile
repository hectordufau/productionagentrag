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

evaluate-retrieval:
	$(PYTHON) -m pytest tests/unit/test_m2a_retrieval.py tests/unit/test_qdrant.py -q
	$(PYTHON) scripts/run_benchmark.py

benchmark-retrieval:
	$(PYTHON) scripts/run_benchmark.py

benchmark: benchmark-retrieval
