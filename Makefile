.PHONY: help setup preprocess grid backtest live monitor close test lint typecheck clean

help:
	@echo "Available targets:"
	@echo "  setup       - Create virtual environment and install dependencies"
	@echo "  preprocess  - Download data and build windows"
	@echo "  grid        - Run parameter grid search (X, Y, Z optimization)"
	@echo "  backtest    - Run walk-forward backtest with current config"
	@echo "  live        - Generate live signals for today"
	@echo "  monitor     - Monitor open positions and generate alerts"
	@echo "  close       - Close positions based on exit criteria"
	@echo "  test        - Run unit tests with pytest"
	@echo "  lint        - Run code linting with ruff"
	@echo "  typecheck   - Run mypy type checking on src/"
	@echo "  clean       - Remove generated files and caches"

setup:
	python -m venv .venv
	.venv/Scripts/pip install -r requirements.txt  # Windows
	# .venv/bin/pip install -r requirements.txt    # Unix/Mac
	@echo "Setup complete! Activate with: .venv\\Scripts\\activate"

preprocess:
	python scripts/preprocess.py

grid:
	python scripts/gridsearch.py

backtest:
	python scripts/backtest.py

live:
	python scripts/live.py

monitor:
	python scripts/monitor.py

close:
	python scripts/close_positions.py

test:
	pytest -q

lint:
	ruff check src/ tests/ scripts/

typecheck:
	mypy src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned cache files"
