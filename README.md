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
source venv/bin/activate  # On Windows: venv\Scripts\activate

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