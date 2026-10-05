.PHONY: help install install-gpu install-worker install-dev test lint start stop status submit

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =========================================================================
# Installation
# =========================================================================

install:  ## Install base package
	pip install -e .

install-gpu:  ## Install with GPU support (for Docling service)
	pip install -e ".[gpu]"

install-worker:  ## Install for Celery workers
	pip install -e ".[worker]"

install-dev:  ## Install with dev tools
	pip install -e ".[dev]"

# =========================================================================
# Services
# =========================================================================

start:  ## Start all services
	./scripts/start_all.sh

stop:  ## Stop all services
	./scripts/stop_all.sh

status:  ## Check service status
	./scripts/start_docling_instances.sh status
	@echo ""
	./scripts/start_worker.sh status

docling-start:  ## Start Docling instances
	./scripts/start_docling_instances.sh start

docling-stop:  ## Stop Docling instances
	./scripts/start_docling_instances.sh stop

worker-start:  ## Start Celery worker
	./scripts/start_worker.sh start

worker-stop:  ## Stop Celery worker
	./scripts/start_worker.sh stop

watchdog:  ## Start Docling watchdog
	./scripts/docling_watchdog.sh

# =========================================================================
# Batch Processing
# =========================================================================

submit:  ## Submit batch: make submit DIR=/path/to/pdfs COLLECTION=name
	meridian submit $(DIR) --collection $(COLLECTION)

batch-status:  ## Check batch status: make batch-status BATCH=batch_id
	meridian status $(BATCH)

batch-list:  ## List all batches
	meridian list

# =========================================================================
# Docker
# =========================================================================

docker-up:  ## Start Docker services
	docker compose up -d

docker-down:  ## Stop Docker services
	docker compose down

docker-build:  ## Build Docker images
	docker compose build

docker-logs:  ## Follow Docker logs
	docker compose logs -f

# =========================================================================
# Development
# =========================================================================

test:  ## Run tests
	pytest tests/ -v

lint:  ## Run linters
	ruff check meridian/
	ruff format --check meridian/

format:  ## Format code
	ruff format meridian/

clean:  ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
