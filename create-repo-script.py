#!/usr/bin/env python3
"""
Script to create the complete crypto momentum backtest repository structure.
Run this script to generate all files and directories.
"""

import os
from pathlib import Path
import textwrap

def create_file(filepath, content):
    """Create a file with the given content."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove leading whitespace while preserving relative indentation
    content = textwrap.dedent(content).strip()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Created: {filepath}")

def create_repository():
    """Create the complete repository structure."""
    
    # Create .gitignore
    create_file('.gitignore', '''
    # Python
    __pycache__/
    *.py[cod]
    *$py.class
    *.so
    .Python
    env/
    venv/
    ENV/
    
    # Data
    data/
    *.json.gz
    *.csv
    
    # Output
    output/
    results/
    logs/
    *.png
    *.jpg
    
    # IDE
    .idea/
    .vscode/
    *.swp
    *.swo
    
    # OS
    .DS_Store
    Thumbs.db
    
    # Jupyter
    .ipynb_checkpoints/
    *.ipynb
    
    # Testing
    .pytest_cache/
    .coverage
    htmlcov/
    
    # Distribution
    dist/
    build/
    *.egg-info/
    ''')
    
    # Create README.md
    create_file('README.md', '''
    # Crypto Momentum Backtesting Engine
    
    A production-grade backtesting system for momentum-based cryptocurrency trading strategies using VectorBT.
    
    ## Features
    
    - **Dynamic Universe Selection**: Top 10-20 coins by market cap with survivorship bias handling
    - **Momentum Signals**: ADX + EWMA crossover with volume confirmation
    - **Risk Parity**: Equal Risk Contribution (ERC) portfolio optimization
    - **Realistic Costs**: Exchange fees, spreads, slippage, and funding rates
    - **Risk Management**: Drawdown limits, correlation monitoring, regime detection
    - **Validation**: Walk-forward analysis, Monte Carlo simulation, parameter sensitivity
    
    ## Installation
    
    ```bash
    # Clone the repository
    git clone https://github.com/yourusername/crypto-momentum-backtest.git
    cd crypto-momentum-backtest
    
    # Create virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate
    
    # Install dependencies
    pip install -r requirements.txt
    ```
    
    ## Quick Start
    
    ```bash
    # Run backtest with default settings
    python -m crypto_momentum_backtest.main
    
    # Fetch latest data and run full validation
    python -m crypto_momentum_backtest.main --fetch-data --validate
    
    # Custom date range
    python -m crypto_momentum_backtest.main --start-date 2022-01-01 --end-date 2023-12-31
    ```
    
    ## Configuration
    
    Edit `config.yaml` to adjust strategy parameters:
    
    ```yaml
    strategy:
      universe_size: 20
      rebalance_frequency: "monthly"
      max_position_size: 0.20
    
    signals:
      adx_threshold: 20
      ewma_fast: 20
      ewma_slow: 50
    ```
    
    ## Project Structure
    
    ```
    crypto_momentum_backtest/
    ├── data/               # Data management
    ├── signals/            # Signal generation
    ├── portfolio/          # Portfolio optimization
    ├── risk/               # Risk management
    ├── costs/              # Transaction costs
    ├── backtest/           # Backtesting engine
    └── utils/              # Utilities
    ```
    
    ## Output
    
    Results are saved to the `output/` directory:
    - `portfolio_values.csv`: Daily portfolio values
    - `trades.csv`: All executed trades
    - `metrics.csv`: Performance metrics
    - `validation_report.json`: Robustness test results
    - Various visualization charts
    
    ## License
    
    MIT License - see LICENSE file for details.
    ''')
    
    # Create requirements.txt
    create_file('requirements.txt', '''
    # Core dependencies
    pandas>=1.5.0
    numpy>=1.23.0
    scipy>=1.9.0
    scikit-learn>=1.1.0
    
    # Backtesting
    vectorbt>=0.24.0
    empyrical>=0.5.5
    
    # Data fetching
    aiohttp>=3.8.0
    
    # Visualization
    matplotlib>=3.6.0
    seaborn>=0.12.0
    plotly>=5.11.0
    
    # Utils
    pyyaml>=6.0
    numba>=0.56.0
    tqdm>=4.64.0
    
    # Development
    pytest>=7.2.0
    black>=22.10.0
    flake8>=5.0.0
    mypy>=0.990
    ''')
    
    # Create setup.py
    create_file('setup.py', '''
    from setuptools import setup, find_packages
    
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
    
    setup(
        name="crypto-momentum-backtest",
        version="1.0.0",
        author="Your Name",
        author_email="your.email@example.com",
        description="Production-grade crypto momentum backtesting system",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/yourusername/crypto-momentum-backtest",
        packages=find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Development Status :: 4 - Beta",
            "Intended Audience :: Financial and Insurance Industry",
            "Topic :: Office/Business :: Financial :: Investment",
        ],
        python_requires=">=3.8",
        install_requires=[
            "pandas>=1.5.0",
            "numpy>=1.23.0",
            "scipy>=1.9.0",
            "scikit-learn>=1.1.0",
            "vectorbt>=0.24.0",
            "empyrical>=0.5.5",
            "aiohttp>=3.8.0",
            "matplotlib>=3.6.0",
            "seaborn>=0.12.0",
            "pyyaml>=6.0",
            "numba>=0.56.0",
        ],
    )
    ''')
    
    # Create config.yaml
    create_file('config.yaml', '''
    strategy:
      universe_size: 20
      rebalance_frequency: "monthly"
      max_position_size: 0.20
      long_short: true
    
    signals:
      adx_period: 14
      adx_threshold: 20
      ewma_fast: 20
      ewma_slow: 50
      volume_filter_multiple: 1.5
    
    risk:
      atr_period: 14
      atr_multiplier: 2.0
      max_correlation: 0.8
      volatility_regime_multiplier: 1.5
      max_drawdown_threshold: 0.15
      max_exchange_exposure: 0.40
    
    costs:
      maker_fee: 0.0002
      taker_fee: 0.0004
      base_spread: 0.0001
      funding_lookback_days: 30
    ''')
    
    # Create __init__.py files
    create_file('crypto_momentum_backtest/__init__.py', '''
    """Crypto Momentum Backtesting System."""
    __version__ = "1.0.0"
    ''')
    
    for module in ['data', 'signals', 'portfolio', 'risk', 'costs', 'backtest', 'utils']:
        create_file(f'crypto_momentum_backtest/{module}/__init__.py', 
                   f'"""{module.capitalize()} module."""')
    
    # Create tests directory
    create_file('tests/__init__.py', '')
    
    # Create example test file
    create_file('tests/test_signals.py', '''
    """Tests for signal generation."""
    import pytest
    import pandas as pd
    import numpy as np
    from crypto_momentum_backtest.signals.signal_generator import SignalGenerator
    
    
    def test_signal_generator_initialization():
        """Test SignalGenerator initialization."""
        generator = SignalGenerator(
            adx_period=14,
            adx_threshold=20,
            ewma_fast=20,
            ewma_slow=50
        )
        
        assert generator.adx_period == 14
        assert generator.adx_threshold == 20
        assert generator.ewma_fast == 20
        assert generator.ewma_slow == 50
    
    
    def test_signal_generation():
        """Test basic signal generation."""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.rand(100) * 1000000
        }, index=dates)
        
        generator = SignalGenerator()
        signals = generator.generate_signals(df)
        
        assert 'long_signal' in signals.columns
        assert 'short_signal' in signals.columns
        assert 'position' in signals.columns
        assert len(signals) == len(df)
    ''')
    
    # Create example notebooks directory
    create_file('notebooks/.gitkeep', '')
    
    # Create GitHub Actions workflow
    create_file('.github/workflows/tests.yml', '''
    name: Tests
    
    on: [push, pull_request]
    
    jobs:
      test:
        runs-on: ubuntu-latest
        strategy:
          matrix:
            python-version: [3.8, 3.9, "3.10"]
        
        steps:
        - uses: actions/checkout@v2
        
        - name: Set up Python ${{ matrix.python-version }}
          uses: actions/setup-python@v2
          with:
            python-version: ${{ matrix.python-version }}
        
        - name: Install dependencies
          run: |
            python -m pip install --upgrade pip
            pip install -r requirements.txt
            pip install pytest pytest-cov
        
        - name: Run tests
          run: |
            pytest tests/ -v --cov=crypto_momentum_backtest
        
        - name: Run linting
          run: |
            pip install flake8 black mypy
            flake8 crypto_momentum_backtest/ --max-line-length=100
            black --check crypto_momentum_backtest/
            mypy crypto_momentum_backtest/ --ignore-missing-imports
    ''')
    
    # Create LICENSE
    create_file('LICENSE', '''
    MIT License
    
    Copyright (c) 2024 [Your Name]
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    ''')
    
    # Create Makefile
    create_file('Makefile', '''
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
    ''')
    
    print("\n" + "="*50)
    print("Repository structure created successfully!")
    print("="*50)
    print("\nNext steps:")
    print("1. cd crypto-momentum-backtest")
    print("2. python create_modules.py  # Run the next script to create all module files")
    print("3. pip install -r requirements.txt")
    print("4. python -m crypto_momentum_backtest.main")


if __name__ == "__main__":
    create_repository()
