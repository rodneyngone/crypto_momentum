#!/usr/bin/env python3
"""
Test and compare different mean return signal strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_momentum_backtest.signals.signal_generator import SignalGenerator
from crypto_momentum_backtest.data.json_storage import JsonStorage


def load_test_data(symbol='BTCUSDT', start_date='2022-01-01', end_date='2023-12-31'):
    """Load test data for signal comparison."""
    storage = JsonStorage(Path('data'))
    
    df = storage.load_range(
        symbol=symbol,
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date)
    )
    
    return df


def test_signal_strategies(df):
    """Test different signal generation strategies."""
    
    strategies = {
        "Original Momentum": {
            "use_mean_return_signal": False,
            "adx_threshold": 30,
            "ewma_fast": 10,
            "ewma_slow": 30
        },
        "Mean Return EWM": {
            "use_mean_return_signal": True,
            "mean_return_window": 30,
            "mean_return_type": "ewm",
            "mean_return_threshold": 0,
            "mean_return_ewm_span": 20,
            "combine_signals": "override"
        },
        "Mean Return Simple": {
            "use_mean_return_signal": True,
            "mean_return_window": 30,
            "mean_return_type": "simple",
            "mean_return_threshold": 0,
            "combine_signals": "override"
        },
        "Hybrid AND": {
            "use_mean_return_signal": True,
            "mean_return_window": 30,
            "mean_return_type": "ewm",
            "mean_return_threshold": 0,
            "mean_return_ewm_span": 20,
            "combine_signals": "and",
            "adx_threshold": 25,
            "ewma_fast": 10,
            "ewma_slow": 30
        },
        "Hybrid OR": {
            "use_mean_return_signal": True,
            "mean_return_window": 30,
            "mean_return_type": "ewm",
            "mean_return_threshold": 0,
            "mean_return_ewm_span": 20,
            "combine_signals": "or",
            "adx_threshold": 30,
            "ewma_fast": 10,
            "ewma_slow": 30
        }
    }
    
    results = {}
    
    for name, params in strategies.items():
        print(f"\n📊 Testing {name} Strategy...")
        
        # Create signal generator
        generator = SignalGenerator(**params)
        
        # Generate signals
        signals_df = generator.generate_signals(df.copy())
        
        # Calculate performance
        returns = df['close'].pct_change()
        signal_returns = returns * signals_df['position'].shift(1)
        
        # Store results
        results[name] = {
            'signals_df': signals_df,
            'signal_returns': signal_returns,
            'total_return': (1 + signal_returns).prod() - 1,
            'sharpe': signal_returns.mean() / signal_returns.std() * np.sqrt(252),
            'win_rate': (signal_returns > 0).sum() / (signal_returns != 0).sum(),
            'num_long': signals_df['long_signal'].sum(),
            'num_short': signals_df['short_signal'].sum(),
            'summary': generator.get_signal_summary(signals_df)
        }
        
        print(f"  Total Return: {results[name]['total_return']:.2%}")
        print(f"  Sharpe Ratio: {results[name]['sharpe']:.2f}")
        print(f"  Win Rate: {results[name]['win_rate']:.2%}")
        print(f"  Signals - Long: {results[name]['num_long']}, Short: {results[name]['num_short']}")
    
    return results


def plot_signal_comparison(df, results):
    """Plot comparison of different signal strategies."""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Signal Strategy Comparison', fontsize=16)
    
    # 1. Cumulative returns comparison
    ax = axes[0, 0]
    
    # Buy and hold
    buy_hold = (1 + df['close'].pct_change()).cumprod()
    ax.plot(df.index, buy_hold, label='Buy & Hold', color='gray', alpha=0.7)
    
    # Strategy returns
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    for (name, result), color in zip(results.items(), colors):
        cum_returns = (1 + result['signal_returns']).cumprod()
        ax.plot(df.index, cum_returns, label=name, color=color)
    
    ax.set_title('Cumulative Returns')
    ax.set_ylabel('Cumulative Return')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Signal counts
    ax = axes[0, 1]
    strategies = list(results.keys())
    long_counts = [results[s]['num_long'] for s in strategies]
    short_counts = [results[s]['num_short'] for s in strategies]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    ax.bar(x - width/2, long_counts, width, label='Long', color='green', alpha=0.7)
    ax.bar(x + width/2, short_counts, width, label='Short', color='red', alpha=0.7)
    
    ax.set_title('Signal Counts')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Performance metrics
    ax = axes[1, 0]
    metrics_df = pd.DataFrame({
        'Total Return': [results[s]['total_return'] for s in strategies],
        'Sharpe Ratio': [results[s]['sharpe'] for s in strategies],
        'Win Rate': [results[s]['win_rate'] for s in strategies]
    }, index=strategies)
    
    metrics_df.plot(kind='bar', ax=ax)
    ax.set_title('Performance Metrics')
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # 4. Mean return signal example (if available)
    ax = axes[1, 1]
    
    # Find a strategy with mean return signals
    mean_return_strategy = None
    for name, result in results.items():
        if 'mean_return' in result['signals_df'].columns:
            mean_return_strategy = name
            break
    
    if mean_return_strategy:
        signals_df = results[mean_return_strategy]['signals_df']
        
        # Plot mean returns
        ax.plot(df.index, signals_df['mean_return'] * 100, label='Mean Return (%)', color='blue')
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Long Threshold')
        ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Short Threshold')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Mark signals
        long_signals = signals_df[signals_df['long_signal']]
        short_signals = signals_df[signals_df['short_signal']]
        
        if len(long_signals) > 0:
            ax.scatter(long_signals.index, long_signals['mean_return'] * 100, 
                      color='green', marker='^', s=50, alpha=0.7, label='Long Signal')
        if len(short_signals) > 0:
            ax.scatter(short_signals.index, short_signals['mean_return'] * 100, 
                      color='red', marker='v', s=50, alpha=0.7, label='Short Signal')
        
        ax.set_title(f'Mean Return Signals ({mean_return_strategy})')
        ax.set_ylabel('30-Day Mean Return (%)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 5. Drawdown comparison
    ax = axes[2, 0]
    
    for (name, result), color in zip(results.items(), colors):
        cum_returns = (1 + result['signal_returns']).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        ax.plot(df.index, drawdown * 100, label=name, color=color)
    
    ax.set_title('Drawdown Comparison')
    ax.set_ylabel('Drawdown (%)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.fill_between(df.index, 0, -15, alpha=0.1, color='red')
    
    # 6. Rolling Sharpe comparison
    ax = axes[2, 1]
    
    window = 252  # 1 year rolling
    for (name, result), color in zip(results.items(), colors):
        rolling_sharpe = (
            result['signal_returns'].rolling(window).mean() / 
            result['signal_returns'].rolling(window).std() * np.sqrt(252)
        )
        ax.plot(df.index, rolling_sharpe, label=name, color=color)
    
    ax.set_title('Rolling 1-Year Sharpe Ratio')
    ax.set_ylabel('Sharpe Ratio')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.3)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/signal_strategy_comparison.png', dpi=300)
    print(f"\n📊 Saved comparison plot to output/signal_strategy_comparison.png")


def main():
    """Run signal strategy comparison."""
    print("🔍 Mean Return Signal Strategy Test")
    print("=" * 60)
    
    # Load test data
    print("\n📊 Loading BTC data...")
    df = load_test_data('BTCUSDT', '2022-01-01', '2023-12-31')
    
    if df is None or df.empty:
        print("❌ No data found. Please run data fetching first.")
        return
    
    print(f"Loaded {len(df)} days of data")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Test strategies
    results = test_signal_strategies(df)
    
    # Plot comparison
    plot_signal_comparison(df, results)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    # Find best strategy
    best_return = max(results.items(), key=lambda x: x[1]['total_return'])
    best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe'])
    
    print(f"\n🏆 Best Total Return: {best_return[0]} ({best_return[1]['total_return']:.2%})")
    print(f"🏆 Best Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1]['sharpe']:.2f})")
    
    # Buy and hold comparison
    buy_hold_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
    print(f"\n📊 Buy & Hold Return: {buy_hold_return:.2%}")
    
    print("\n💡 Key Insights:")
    print("- Mean return signals can capture different market dynamics")
    print("- EWM gives more weight to recent returns vs simple average")
    print("- Combining signals (AND/OR) affects trade frequency and quality")
    print("- Consider market regime when choosing signal type")


if __name__ == "__main__":
    main()