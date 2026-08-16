.PHONY: install install-dev test lint format type-check run clean help

help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -r requirements-dev.txt

test:  ## Run tests with pytest
	pytest tests/ -v --cov=src/weather_pipeline --cov-report=term-missing

lint:  ## Run ruff linter
	ruff check src/ tests/

format:  ## Format code with ruff
	ruff format src/ tests/

type-check:  ## Run mypy type checker
	mypy src/weather_pipeline --ignore-missing-imports

run:  ## Run the pipeline with default config
	python -m weather_pipeline.cli --config configs/default.yaml --stage all

clean:  ## Clean up generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
