.PHONY: install dev test lint format run-api run-reranker docker-up docker-down

install:
	pip install -e ".[dev]"

dev:
	pip install -e ".[dev,reranker]"

test:
	pytest tests/ -v

lint:
	ruff check .

format:
	ruff check --fix .
	ruff format .

run-api:
	uvicorn oracle.api.app:app --host 0.0.0.0 --port 8010 --reload

run-reranker:
	python reranker/main.py

docker-up:
	docker compose up -d

docker-down:
	docker compose down
