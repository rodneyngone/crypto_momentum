#!/usr/bin/env python3
"""
Compatibility wrapper to ensure enhanced components work with existing code.
This file maps old component names to enhanced versions.
"""

# Signal components
from crypto_momentum_backtest.signals import (
    SignalGenerator,  # This is now EnhancedSignalGenerator
    SignalType,
    SignalMetrics
)

# Portfolio components
from crypto_momentum_backtest.portfolio import (
    ERCOptimizer,  # This is now EnhancedERCOptimizer
    Rebalancer     # This is now EnhancedRebalancer
)

# Risk components
from crypto_momentum_backtest.risk import (
    RiskManager,   # This is now EnhancedRiskManager
    MarketRegime
)

# Data components
from crypto_momentum_backtest.data import (
    UniverseManager  # This is now EnhancedUniverseManager
)

print("✅ Using enhanced components as defaults")

# Verify enhanced features are available
def verify_enhancements():
    """Quick check that enhanced features are available."""
    checks = {
        'SignalGenerator.generate_signals': hasattr(SignalGenerator, 'generate_signals'),
        'ERCOptimizer.concentration_mode': hasattr(ERCOptimizer, '__init__') and 'concentration_mode' in ERCOptimizer.__init__.__code__.co_varnames,
        'RiskManager.detect_market_regime': hasattr(RiskManager, 'detect_market_regime'),
        'Rebalancer.momentum_preservation': hasattr(Rebalancer, '__init__') and 'momentum_preservation' in Rebalancer.__init__.__code__.co_varnames,
    }
    
    print("\nEnhancement verification:")
    for feature, available in checks.items():
        status = "✅" if available else "❌"
        print(f"  {status} {feature}")
    
    return all(checks.values())

if __name__ == "__main__":
    if verify_enhancements():
        print("\n✅ All enhanced features are available!")
    else:
        print("\n❌ Some enhanced features missing - check imports")
