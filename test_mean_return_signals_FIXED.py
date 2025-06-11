#!/usr/bin/env python3
"""
FIXED VERSION: Mean Return Signal Strategy Test with proper thresholds for crypto
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_btc_data():
    """Load BTC data"""
    data_files = ['data/BTCUSDT.csv', 'BTCUSDT.csv', 'data/btc_data.csv']
    
    for file_path in data_files:
        if Path(file_path).exists():
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            print(f"Loaded {len(data)} days of data")
            print(f"Date range: {data.index[0]} to {data.index[-1]}")
            return data
    
    raise FileNotFoundError("No BTC data file found")

def test_momentum_strategy(data, threshold=0.01):
    """Test original momentum strategy"""
    data = data.copy()
    data['returns'] = data['close'].pct_change()
    
    # Simple momentum
    data['ewma'] = data['close'].ewm(span=20).mean()
    data['momentum'] = (data['close'] / data['ewma'] - 1)
    
    long_signals = data['momentum'] > threshold
    short_signals = data['momentum'] < -threshold
    
    return long_signals, short_signals, data['returns']

def test_mean_return_ewm_strategy(data, span=5, threshold=0.002):
    """Test EWM mean return strategy with FIXED threshold"""
    data = data.copy()
    data['returns'] = data['close'].pct_change()
    
    # EWM of returns (not prices!)
    ewm_returns = data['returns'].ewm(span=span).mean()
    
    long_signals = ewm_returns > threshold
    short_signals = ewm_returns < -threshold
    
    print(f"EWM Strategy - Span: {span}, Threshold: {threshold}")
    print(f"  EWM returns range: {ewm_returns.min():.6f} to {ewm_returns.max():.6f}")
    print(f"  Signals: Long={long_signals.sum()}, Short={short_signals.sum()}")
    
    return long_signals, short_signals, data['returns']

def test_mean_return_simple_strategy(data, window=3, threshold=0.003):
    """Test simple mean return strategy with FIXED threshold"""
    data = data.copy()
    data['returns'] = data['close'].pct_change()
    
    # Rolling mean of returns
    simple_mean = data['returns'].rolling(window=window).mean()
    
    long_signals = simple_mean > threshold
    short_signals = simple_mean < -threshold
    
    print(f"Simple Strategy - Window: {window}, Threshold: {threshold}")
    print(f"  Simple mean range: {simple_mean.min():.6f} to {simple_mean.max():.6f}")
    print(f"  Signals: Long={long_signals.sum()}, Short={short_signals.sum()}")
    
    return long_signals, short_signals, data['returns']

def calculate_performance(long_signals, short_signals, returns):
    """Calculate strategy performance"""
    positions = long_signals.astype(int) - short_signals.astype(int)
    strategy_returns = positions.shift(1) * returns
    strategy_returns = strategy_returns.dropna()
    
    if len(strategy_returns) == 0 or strategy_returns.std() == 0:
        return {
            'total_return': 0,
            'sharpe': 0,
            'win_rate': 0,
            'signals': {'long': 0, 'short': 0}
        }
    
    total_return = (1 + strategy_returns).prod() - 1
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    win_rate = (strategy_returns > 0).mean()
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'signals': {'long': long_signals.sum(), 'short': short_signals.sum()}
    }

def main():
    """Main test function with FIXED parameters"""
    print("FIXED Mean Return Signal Strategy Test")
    print("="*60)
    
    # Load data
    data = load_btc_data()
    
    # Test strategies with FIXED thresholds
    print("\nTesting Original Momentum Strategy...")
    long_mom, short_mom, returns = test_momentum_strategy(data, threshold=0.01)
    mom_perf = calculate_performance(long_mom, short_mom, returns)
    print(f"  Total Return: {mom_perf['total_return']:.2%}")
    print(f"  Sharpe Ratio: {mom_perf['sharpe']:.2f}")
    print(f"  Win Rate: {mom_perf['win_rate']:.2%}")
    print(f"  Signals - Long: {mom_perf['signals']['long']}, Short: {mom_perf['signals']['short']}")
    
    print("\nTesting FIXED Mean Return EWM Strategy...")
    long_ewm, short_ewm, returns = test_mean_return_ewm_strategy(data, span=5, threshold=0.002)
    ewm_perf = calculate_performance(long_ewm, short_ewm, returns)
    print(f"  Total Return: {ewm_perf['total_return']:.2%}")
    print(f"  Sharpe Ratio: {ewm_perf['sharpe']:.2f}")
    print(f"  Win Rate: {ewm_perf['win_rate']:.2%}")
    print(f"  Signals - Long: {ewm_perf['signals']['long']}, Short: {ewm_perf['signals']['short']}")
    
    print("\nTesting FIXED Simple Mean Return Strategy...")
    long_simple, short_simple, returns = test_mean_return_simple_strategy(data, window=3, threshold=0.003)
    simple_perf = calculate_performance(long_simple, short_simple, returns)
    print(f"  Total Return: {simple_perf['total_return']:.2%}")
    print(f"  Sharpe Ratio: {simple_perf['sharpe']:.2f}")
    print(f"  Win Rate: {simple_perf['win_rate']:.2%}")
    print(f"  Signals - Long: {simple_perf['signals']['long']}, Short: {simple_perf['signals']['short']}")
    
    print("\n" + "="*60)
    print("FIXED TEST COMPLETE!")
    print("="*60)
    print("If these strategies now generate signals, the issue was threshold calibration.")

if __name__ == "__main__":
    main()
