#!/usr/bin/env python3
"""
Test script to verify signal generation is working.
"""

import pandas as pd
import numpy as np
import yaml

def test_signals():
    """Test if signals are being generated"""
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Create sample data
    dates = pd.date_range('2022-01-01', '2022-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate realistic crypto returns
    returns = np.random.normal(0.001, 0.04, len(dates))
    prices = [40000]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'close': prices,
        'returns': [0] + list(np.diff(prices) / prices[:-1])
    }, index=dates)
    
    # Test momentum signals
    ewm_span = config['signals']['ewm_span']
    threshold = config['signals']['threshold']
    
    data['ewma'] = data['close'].ewm(span=ewm_span).mean()
    data['momentum'] = (data['close'] / data['ewma'] - 1)
    
    long_momentum = (data['momentum'] > threshold).sum()
    short_momentum = (data['momentum'] < -threshold).sum()
    
    print(f"SIGNAL TEST RESULTS:")
    print(f"Momentum Strategy (threshold={threshold}):")
    print(f"  Long signals: {long_momentum}")
    print(f"  Short signals: {short_momentum}")
    print(f"  Total signals: {long_momentum + short_momentum}")
    
    # Test mean return signals
    mean_threshold = config['signals'].get('mean_return_threshold', 0.005)
    ewm_returns = data['returns'].ewm(span=5).mean()
    
    long_mean = (ewm_returns > mean_threshold).sum()
    short_mean = (ewm_returns < -mean_threshold).sum()
    
    print(f"\nMean Return EWM Strategy (threshold={mean_threshold}):")
    print(f"  Long signals: {long_mean}")
    print(f"  Short signals: {short_mean}")
    print(f"  Total signals: {long_mean + short_mean}")
    
    # Check if fix worked
    total_signals = long_momentum + short_momentum + long_mean + short_mean
    if total_signals > 0:
        print(f"\nSUCCESS: Signal generation is working!")
        print(f"Total signals generated: {total_signals}")
    else:
        print(f"\nISSUE: Still generating 0 signals")
        print("Try reducing thresholds further")

if __name__ == "__main__":
    test_signals()
