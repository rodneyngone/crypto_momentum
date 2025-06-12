"""Main entry point for the crypto momentum backtesting system."""
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio
import sys
import os

# Add parent directory to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports when running as a script
# try:
from crypto_momentum_backtest.utils.config import Config
from crypto_momentum_backtest.utils.logger import setup_logger
from crypto_momentum_backtest.data.binance_fetcher import BinanceFetcher
from crypto_momentum_backtest.data.universe_manager_enhanced import EnhancedUniverseManager
from crypto_momentum_backtest.backtest.engine import BacktestEngine
from crypto_momentum_backtest.backtest.validator import BacktestValidator
from crypto_momentum_backtest.backtest.metrics import MetricsCalculator
# except ImportError:
#     # Fallback to relative imports if running as module
#     from utils.config import Config
#     from utils.logger import setup_logger
#     from data.binance_fetcher import BinanceFetcher
#     from data.universe_manager import UniverseManager
#     from backtest.engine import BacktestEngine
#     from backtest.validator import BacktestValidator
#     from backtest.metrics import MetricsCalculator


def setup_environment(args):
    """Set up the environment and configuration."""
    # Create directories
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logger(
        name="crypto_backtest",
        level=args.log_level,
        log_dir=output_dir / "logs"
    )
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.suffix == '.yaml':
        config = Config.from_yaml(config_path)
    else:
        config = Config.from_json(config_path)
    
    return config, logger, data_dir, output_dir


async def fetch_data(config, data_dir, symbols, start_date, end_date, logger):
    """Fetch historical data if needed."""
    fetcher = BinanceFetcher(data_dir, logger=logger)
    
    logger.info(f"Fetching data for {len(symbols)} symbols")
    
    await fetcher.update_universe_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        max_concurrent=5
    )
    
    logger.info("Data fetching complete")


def run_backtest(config, data_dir, start_date, end_date, logger):
    """Run the main backtest."""
    # Initialize backtest engine
    engine = BacktestEngine(config, data_dir, logger=logger)
    
    # Run backtest
    results = engine.run_backtest(
        start_date=start_date,
        end_date=end_date,
        initial_capital=1_000_000
    )
    
    return results


def validate_results(engine, results, start_date, end_date, logger):
    """Validate backtest results."""
    validator = BacktestValidator(logger=logger)
    
    # Validation checks
    validation_checks = validator.validate_results(results)
    
    # Walk-forward analysis
    logger.info("Running walk-forward analysis...")
    walk_forward_results = validator.walk_forward_analysis(
        backtest_func=engine.run_backtest,
        start_date=start_date,
        end_date=end_date,
        initial_capital=1_000_000
    )
    
    # Monte Carlo simulation
    logger.info("Running Monte Carlo simulation...")
    monte_carlo_results = validator.monte_carlo_simulation(
        returns=results['returns'],
        n_simulations=1000
    )
    
    # Parameter sensitivity
    param_ranges = {
        'adx_threshold': [15, 20, 25, 30],
        'ewma_fast': [10, 15, 20, 25],
        'ewma_slow': [40, 50, 60, 70]
    }
    
    logger.info("Running parameter sensitivity analysis...")
    sensitivity_results = validator.parameter_sensitivity(
        backtest_func=engine.run_backtest,
        base_params={
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': 1_000_000
        },
        param_ranges=param_ranges
    )
    
    # Generate validation report
    validation_report = validator.generate_validation_report(
        walk_forward_results=walk_forward_results,
        monte_carlo_results=monte_carlo_results,
        sensitivity_results=sensitivity_results,
        validation_checks=validation_checks
    )
    
    return validation_report


def generate_visualizations(results, output_dir):
    """Generate visualization plots."""
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Portfolio value over time
    fig, ax = plt.subplots(figsize=(12, 6))
    results['portfolio'].plot(ax=ax, linewidth=2)
    ax.set_title('Portfolio Value Over Time', fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    plt.savefig(output_dir / 'portfolio_value.png', dpi=300)
    plt.close()
    
    # 2. Drawdown chart
    cum_returns = (1 + results['returns']).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    
    fig, ax = plt.subplots(figsize=(12, 6))
    drawdown.plot(ax=ax, linewidth=2, color='red', alpha=0.7)
    ax.fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.3)
    ax.set_title('Portfolio Drawdown', fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.tight_layout()
    plt.savefig(output_dir / 'drawdown.png', dpi=300)
    plt.close()
    
    # 3. Monthly returns heatmap
    monthly_returns = results['returns'].resample('ME').apply(
        lambda x: (1 + x).prod() - 1
    )
    
    # Reshape for heatmap
    monthly_returns_df = pd.DataFrame(
        monthly_returns.values,
        index=monthly_returns.index,
        columns=['Returns']
    )
    monthly_returns_df['Year'] = monthly_returns_df.index.year
    monthly_returns_df['Month'] = monthly_returns_df.index.month
    
    pivot_monthly = monthly_returns_df.pivot(
        index='Year',
        columns='Month',
        values='Returns'
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        pivot_monthly,
        annot=True,
        fmt='.1%',
        cmap='RdYlGn',
        center=0,
        ax=ax,
        cbar_kws={'label': 'Monthly Return'}
    )
    ax.set_title('Monthly Returns Heatmap', fontsize=16)
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    plt.tight_layout()
    plt.savefig(output_dir / 'monthly_returns_heatmap.png', dpi=300)
    plt.close()
    
    # 4. Position distribution
    if 'positions' in results and not results['positions'].empty:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Number of positions over time
        n_positions = (results['positions'] != 0).sum(axis=1)
        n_positions.plot(ax=ax1, linewidth=2)
        ax1.set_title('Number of Active Positions', fontsize=14)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Number of Positions')
        
        # Position sizes distribution
        position_sizes = results['positions'][results['positions'] != 0].values.flatten()
        ax2.hist(position_sizes, bins=50, alpha=0.7, edgecolor='black')
        ax2.set_title('Position Size Distribution', fontsize=14)
        ax2.set_xlabel('Position Size')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'positions.png', dpi=300)
        plt.close()


def save_results(results, metrics, validation_report, output_dir):
    """Save results to files."""
    # Save portfolio values
    results['portfolio'].to_csv(output_dir / 'portfolio_values.csv')
    
    # Save trades
    if 'trades' in results and not results['trades'].empty:
        results['trades'].to_csv(output_dir / 'trades.csv', index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.to_csv(output_dir / 'metrics.csv')
    
    # Save validation report
    import json
    with open(output_dir / 'validation_report.json', 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)
    
    # Generate summary report
    generate_summary_report(results, metrics, validation_report, output_dir)


def generate_summary_report(results, metrics, validation_report, output_dir):
    """Generate a summary report."""
    report_lines = [
        "# Crypto Momentum Backtest Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Performance Summary",
        f"- Total Return: {metrics['returns']['total_return']:.2%}",
        f"- Annual Return: {metrics['returns']['annual_return']:.2%}",
        f"- Sharpe Ratio: {metrics['returns']['sharpe_ratio']:.2f}",
        f"- Max Drawdown: {metrics['returns']['max_drawdown']:.2%}",
        f"- Win Rate: {metrics['returns']['win_rate']:.2%}",
        "\n## Risk Metrics",
        f"- Annual Volatility: {metrics['returns']['annual_volatility']:.2%}",
        f"- Sortino Ratio: {metrics['returns']['sortino_ratio']:.2f}",
        f"- Calmar Ratio: {metrics['returns']['calmar_ratio']:.2f}",
        "\n## Trading Statistics",
        f"- Total Trades: {metrics['trades']['total_trades']}",
        f"- Average Trade Size: ${metrics['trades']['avg_trade_size']:,.0f}",
        f"- Unique Symbols: {metrics['trades']['unique_symbols_traded']}",
        "\n## Validation Results",
        f"- All Checks Passed: {validation_report.get('all_checks_passed', 'N/A')}",
        f"- Walk-Forward Sharpe Degradation: {validation_report.get('walk_forward', {}).get('sharpe_degradation', 0):.2f}",
        f"- Monte Carlo VaR (5%): {validation_report.get('monte_carlo', {}).get('value_at_risk_5%', 0):.2%}",
        "\n## Files Generated",
        "- portfolio_values.csv: Daily portfolio values",
        "- trades.csv: All executed trades",
        "- metrics.csv: Detailed performance metrics",
        "- validation_report.json: Complete validation results",
        "- *.png: Various visualization charts"
    ]
    
    with open(output_dir / 'summary_report.md', 'w') as f:
        f.write('\n'.join(report_lines))


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Crypto Momentum Backtesting System'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Data directory path'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory path'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2022-01-01',
        help='Backtest start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default='2023-12-31',
        help='Backtest end date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--fetch-data',
        action='store_true',
        help='Fetch latest data before backtesting'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        default=True,
        help='Run validation and robustness tests'
    )

    parser.add_argument(
        '--no-validate',
        action='store_false',
        dest='validate',
        help='Skip validation and robustness tests'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Set up environment
    config, logger, data_dir, output_dir = setup_environment(args)
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    try:
        # Get universe
        universe_manager = EnhancedUniverseManager(
            data_dir,
            universe_size=config.strategy.universe_size,
            logger=logger
        )
        
        symbols = universe_manager.get_universe(start_date)
        
        # Fetch data if requested
        if args.fetch_data:
            logger.info("Fetching latest market data...")
            asyncio.run(fetch_data(
                config, data_dir, symbols, start_date, end_date, logger
            ))
        
        # Run backtest
        logger.info("Running backtest...")
        results = run_backtest(config, data_dir, start_date, end_date, logger)
        
        # Calculate metrics
        logger.info("Calculating performance metrics...")
        metrics = MetricsCalculator.calculate_all_metrics(
            portfolio_values=results['portfolio'],
            returns=results['returns'],
            positions=results['positions'],
            trades=results['trades']
        )
        
        # Validate results if requested
        validation_report = {}
        if args.validate:
            logger.info("Validating results...")
            engine = BacktestEngine(config, data_dir, logger=logger)
            validation_report = validate_results(
                engine, results, start_date, end_date, logger
            )
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        generate_visualizations(results, output_dir)
        
        # Save results
        logger.info("Saving results...")
        save_results(results, metrics, validation_report, output_dir)
        
        # Print summary
        print("\n" + "="*50)
        print("BACKTEST COMPLETE")
        print("="*50)
        print(f"Total Return: {metrics['returns']['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['returns']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['returns']['max_drawdown']:.2%}")
        print(f"\nResults saved to: {output_dir}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
