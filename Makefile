.PHONY: install test lint format clean run

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v --cov=crypto_momentum_backtest

lint:
	flake8 crypto_momentum_backtest/ --max-line-length=100
	mypy crypto_momentum_backtest/ --ignore-missing-imports

format:
	black crypto_momentum_backtest/
	isort crypto_momentum_backtest/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf .coverage htmlcov/
	rm -rf output/ logs/

run:
	python -m crypto_momentum_backtest.main

run-full:
	python -m crypto_momentum_backtest.main --fetch-data --validate