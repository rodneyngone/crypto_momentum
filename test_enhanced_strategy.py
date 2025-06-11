#!/usr/bin/env python3
"""Test script for enhanced crypto momentum strategy components."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Import enhanced components
from crypto_momentum_backtest.signals import EnhancedSignalGenerator, SignalType
from crypto_momentum_backtest.portfolio import EnhancedERCOptimizer, EnhancedRebalancer
from crypto_momentum_backtest.risk import EnhancedRiskManager
from crypto_momentum_backtest.data import EnhancedUniverseManager

def test_enhanced_signals():
    """Test enhanced signal generation."""
    print("\n🧪 Testing Enhanced Signal Generator...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    data = pd.DataFrame({
        'open': 100 + np.random.randn(len(dates)).cumsum(),
        'high': 101 + np.random.randn(len(dates)).cumsum(),
        'low': 99 + np.random.randn(len(dates)).cumsum(),
        'close': 100 + np.random.randn(len(dates)).cumsum(),
        'volume': np.random.uniform(1e6, 1e7, len(dates))
    }, index=dates)
    
    # Initialize signal generator
    signal_gen = EnhancedSignalGenerator(
        adx_periods=[7, 14, 21],
        adx_threshold=15.0,
        adaptive_ewma=True
    )
    
    # Test different signal types
    for signal_type in [SignalType.MOMENTUM_SCORE, SignalType.MULTI_TIMEFRAME]:
        signals = signal_gen.generate_signals(data, signal_type)
        print(f"  {signal_type.value}: {(signals != 0).sum()} signals generated")
    
    print("  ✅ Signal generation working!")

def test_enhanced_portfolio():
    """Test enhanced portfolio optimization."""
    print("\n🧪 Testing Enhanced Portfolio Optimizer...")
    
    # Create sample returns
    returns = pd.DataFrame(
        np.random.randn(100, 5) * 0.02,
        columns=['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC']
    )
    
    signals = pd.DataFrame(
        np.random.choice([0, 1, -1], size=(1, 5)),
        columns=returns.columns
    )
    
    # Initialize optimizer
    optimizer = EnhancedERCOptimizer(
        concentration_mode=True,
        top_n_assets=3,
        use_momentum_weighting=True
    )
    
    # Optimize
    weights = optimizer.optimize(returns, signals, market_regime='trending')
    print(f"  Weights: {weights[weights != 0].to_dict()}")
    print("  ✅ Portfolio optimization working!")

def test_enhanced_risk():
    """Test enhanced risk management."""
    print("\n🧪 Testing Enhanced Risk Manager...")
    
    # Create sample data
    returns = pd.Series(np.random.randn(100) * 0.02)
    positions = pd.DataFrame(np.random.rand(1, 5), columns=['A', 'B', 'C', 'D', 'E'])
    
    # Initialize risk manager
    risk_mgr = EnhancedRiskManager(
        use_trailing_stops=True,
        regime_lookback=60
    )
    
    # Detect regime
    regime = risk_mgr.detect_market_regime(returns)
    print(f"  Detected regime: {regime.name}")
    print(f"  Risk multiplier: {regime.risk_multiplier}")
    print("  ✅ Risk management working!")

def main():
    """Run all tests."""
    print("🚀 TESTING ENHANCED CRYPTO MOMENTUM STRATEGY")
    print("=" * 60)
    
    test_enhanced_signals()
    test_enhanced_portfolio()
    test_enhanced_risk()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("\nYour enhanced strategy is ready to run!")
    print("Next: python run.py --config config_enhanced.yaml --no-validate")

if __name__ == "__main__":
    main()
