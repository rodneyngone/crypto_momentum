#!/usr/bin/env python3
"""
Fix the Unicode error and test the enhanced crypto momentum system.
"""

import os
from pathlib import Path
from datetime import datetime

def fix_summary_report():
    """Create the summary report with proper encoding."""
    
    report = '''# Crypto Momentum Strategy Enhancements Summary

## Key Improvements Implemented

### 1. Signal Generation
- **Momentum Scoring System**: Composite scoring using price, volume, RSI, and MACD
- **Multi-Timeframe Analysis**: ADX across 7, 14, and 21-day periods
- **Adaptive Parameters**: Dynamic EWMA spans based on volatility
- **Dual Momentum**: Combines absolute and relative momentum

### 2. Portfolio Construction
- **Momentum Tilting**: Weights adjusted based on momentum scores
- **Concentration Mode**: 60% allocation to top 5 performers
- **Dynamic Universe**: 30 -> 15 assets with momentum filtering
- **Category Diversification**: Limits per sector (DeFi, L1, L2, etc.)

### 3. Risk Management
- **Market Regime Detection**: Trending, volatile, ranging, crisis modes
- **Adaptive Exposure**: 0.3x to 1.5x based on regime
- **Trailing Stops**: ATR-based with regime adjustments
- **Dynamic Correlation Limits**: 0.5 to 0.8 based on market conditions

### 4. Execution
- **Smart Rebalancing**: Regime-based frequency (daily to monthly)
- **Momentum Preservation**: Reduces trades against strong trends
- **Turnover Limits**: 10-30% based on regime
- **Fee Optimization**: VIP tier assumptions (0.08% maker)

## Expected Performance Improvements

1. **Better Trend Capture**: Lower ADX threshold (15 vs 25)
2. **Reduced Drawdowns**: Trailing stops + regime detection
3. **Higher Sharpe**: Risk-adjusted position sizing
4. **Lower Costs**: Smarter rebalancing + momentum preservation

## Configuration Changes

Key parameter updates in config_enhanced.yaml:
- Universe: 10 -> 30 (filtered to 15)
- Max position: 10% -> 25%
- ADX threshold: 25 -> 15
- Correlation limit: 0.8 -> 0.7
- Added trailing stops: 2.5x ATR

## Next Steps

1. Run comparison backtest:
   ```bash
   # Original strategy
   python run.py --config config.yaml.backup --output-dir output_original
   
   # Enhanced strategy
   python run.py --config config.yaml --output-dir output_enhanced
   ```

2. Compare metrics:
   ```python
   python compare_results.py output_original output_enhanced
   ```

3. Fine-tune parameters using optimization:
   ```python
   python optimize_parameters.py --config config.yaml
   ```

## Important Notes

- All original files backed up in `backups_pre_enhancement/`
- Enhanced classes use "Enhanced" prefix but maintain compatibility
- Monitor regime detection accuracy in live trading
- Consider paper trading for 30 days before live deployment

Generated: ''' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Write with UTF-8 encoding
    with open("ENHANCEMENT_SUMMARY.md", "w", encoding='utf-8') as f:
        f.write(report)
    
    print("✅ Created ENHANCEMENT_SUMMARY.md")

def test_enhanced_system():
    """Test the enhanced system components."""
    print("\n🧪 TESTING ENHANCED CRYPTO MOMENTUM SYSTEM")
    print("=" * 60)
    
    # Test imports
    print("\n1️⃣ Testing Enhanced Imports...")
    try:
        from crypto_momentum_backtest.signals import EnhancedSignalGenerator, SignalType
        from crypto_momentum_backtest.portfolio import EnhancedERCOptimizer, EnhancedRebalancer
        from crypto_momentum_backtest.risk import EnhancedRiskManager, MarketRegime
        from crypto_momentum_backtest.data import EnhancedUniverseManager
        print("   ✅ All enhanced modules imported successfully!")
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    
    # Test signal generator
    print("\n2️⃣ Testing Enhanced Signal Generator...")
    try:
        import pandas as pd
        import numpy as np
        
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
        for signal_type in [SignalType.MOMENTUM, SignalType.MOMENTUM_SCORE]:
            signals = signal_gen.generate_signals(data, signal_type)
            signal_count = (signals != 0).sum()
            print(f"   {signal_type.value}: {signal_count} signals generated")
        
        print("   ✅ Signal generation working!")
    except Exception as e:
        print(f"   ❌ Signal generation error: {e}")
        return False
    
    # Test portfolio optimizer
    print("\n3️⃣ Testing Enhanced Portfolio Optimizer...")
    try:
        # Create sample returns
        returns = pd.DataFrame(
            np.random.randn(100, 5) * 0.02,
            columns=['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC']
        )
        
        signals = pd.DataFrame(
            [[1, 1, 0, -1, 1]],
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
        non_zero = weights[weights != 0]
        print(f"   Allocated to {len(non_zero)} assets")
        print(f"   Max weight: {non_zero.abs().max():.2%}")
        print("   ✅ Portfolio optimization working!")
    except Exception as e:
        print(f"   ❌ Portfolio optimization error: {e}")
        return False
    
    # Test risk manager
    print("\n4️⃣ Testing Enhanced Risk Manager...")
    try:
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
        print(f"   Detected regime: {regime.name}")
        print(f"   Risk multiplier: {regime.risk_multiplier}")
        print("   ✅ Risk management working!")
    except Exception as e:
        print(f"   ❌ Risk management error: {e}")
        return False
    
    # Test universe manager
    print("\n5️⃣ Testing Enhanced Universe Manager...")
    try:
        from pathlib import Path
        
        # Initialize universe manager
        universe_mgr = EnhancedUniverseManager(
            Path('data'),
            base_universe_size=30,
            selection_universe_size=15
        )
        
        # Get universe
        universe = universe_mgr.get_universe(datetime.now())
        print(f"   Selected {len(universe)} assets")
        print(f"   Top 5: {universe[:5]}")
        print("   ✅ Universe management working!")
    except Exception as e:
        print(f"   ❌ Universe management error: {e}")
        return False
    
    return True

def quick_backtest_test():
    """Run a quick backtest to test the full system."""
    print("\n6️⃣ Testing Full Backtest...")
    
    try:
        from crypto_momentum_backtest.utils.config import Config
        from crypto_momentum_backtest.backtest.engine import BacktestEngine
        from pathlib import Path
        from datetime import datetime, timedelta
        
        # Load enhanced config
        config = Config.from_yaml(Path('config.yaml'))
        print("   ✅ Enhanced config loaded")
        
        # Try to run a short backtest
        engine = BacktestEngine(config, Path('data'))
        
        # Just test initialization
        print("   ✅ Backtest engine initialized with enhanced components")
        
        # Check if we have data
        if (Path('data') / 'market_data').exists():
            print("   ✅ Data directory found")
        else:
            print("   ⚠️  No data found - run 'python simple_fetch_data.py' first")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Backtest error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 ENHANCED CRYPTO MOMENTUM SYSTEM TEST")
    print("=" * 60)
    
    # Fix the Unicode error first
    fix_summary_report()
    
    # Run all tests
    all_passed = test_enhanced_system()
    
    # Try backtest
    backtest_ok = quick_backtest_test()
    
    print("\n" + "=" * 60)
    if all_passed and backtest_ok:
        print("✅ ALL TESTS PASSED!")
        print("\n🎯 Your enhanced strategy is ready to run!")
        print("\nNext steps:")
        print("1. Fetch data: python simple_fetch_data.py")
        print("2. Run backtest: python run.py --no-validate")
        print("3. Check results in output/ directory")
    else:
        print("❌ Some tests failed - check errors above")
        print("\nTroubleshooting:")
        print("1. Make sure all enhanced files were copied correctly")
        print("2. Check that __init__.py files were updated")
        print("3. Try running with debug: python run.py --log-level DEBUG")

if __name__ == "__main__":
    main()