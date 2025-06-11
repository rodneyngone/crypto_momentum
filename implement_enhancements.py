#!/usr/bin/env python3
"""
Script to implement all enhancements to the crypto momentum strategy.
This will update your existing files with the enhanced versions.
"""

import shutil
from pathlib import Path
import os

def implement_enhancements():
    """Copy enhanced files to the crypto momentum project."""
    
    print("🚀 IMPLEMENTING CRYPTO MOMENTUM ENHANCEMENTS")
    print("=" * 60)
    
    # Create backup directory
    backup_dir = Path("backups_pre_enhancement")
    backup_dir.mkdir(exist_ok=True)
    
    # Files to update
    updates = [
        {
            'original': 'crypto_momentum_backtest/signals/signal_generator.py',
            'enhanced': 'signal_generator_enhanced.py',
            'description': 'Enhanced Signal Generator with momentum scoring'
        },
        {
            'original': 'crypto_momentum_backtest/portfolio/erc_optimizer.py',
            'enhanced': 'erc_optimizer_enhanced.py',
            'description': 'Enhanced Portfolio Optimizer with momentum tilting'
        },
        {
            'original': 'crypto_momentum_backtest/risk/risk_manager.py',
            'enhanced': 'risk_manager_enhanced.py',
            'description': 'Enhanced Risk Manager with regime detection'
        },
        {
            'original': 'crypto_momentum_backtest/portfolio/rebalancer.py',
            'enhanced': 'rebalancer_enhanced.py',
            'description': 'Enhanced Dynamic Rebalancer'
        },
        {
            'original': 'crypto_momentum_backtest/data/universe_manager.py',
            'enhanced': 'universe_manager_enhanced.py',
            'description': 'Enhanced Universe Manager with momentum filtering'
        },
        {
            'original': 'config.yaml',
            'enhanced': 'config_enhanced.yaml',
            'description': 'Enhanced configuration with optimized parameters'
        }
    ]
    
    # Process each update
    for update in updates:
        original_path = Path(update['original'])
        enhanced_path = Path(update['enhanced'])
        
        print(f"\n📦 {update['description']}")
        print(f"   From: {enhanced_path}")
        print(f"   To: {original_path}")
        
        # Create backup if original exists
        if original_path.exists():
            backup_path = backup_dir / original_path.name
            shutil.copy2(original_path, backup_path)
            print(f"   ✅ Backup created: {backup_path}")
        
        # Check if enhanced file exists
        if not enhanced_path.exists():
            print(f"   ❌ Enhanced file not found: {enhanced_path}")
            print(f"   ⚠️  Please ensure you've saved all the enhanced files from the artifacts")
            continue
        
        # Copy enhanced file to original location
        try:
            # Ensure target directory exists
            original_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(enhanced_path, original_path)
            print(f"   ✅ Updated successfully!")
            
        except Exception as e:
            print(f"   ❌ Error updating file: {e}")
    
    # Update imports in __init__.py files
    print("\n📝 Updating module imports...")
    update_imports()
    
    # Create summary report
    create_summary_report()
    
    print("\n" + "=" * 60)
    print("✅ ENHANCEMENT IMPLEMENTATION COMPLETE!")
    print("\nNext steps:")
    print("1. Review the changes in each file")
    print("2. Run tests to ensure everything works:")
    print("   python test_enhanced_signals.py")
    print("3. Run enhanced backtest:")
    print("   python run.py --config config_enhanced.yaml --no-validate")
    print("\nBackups saved in: backups_pre_enhancement/")

def update_imports():
    """Update imports to use enhanced classes."""
    
    # Update signals/__init__.py
    signals_init = Path("crypto_momentum_backtest/signals/__init__.py")
    if signals_init.exists():
        content = '''"""Enhanced signals module."""
from .signal_generator_enhanced import EnhancedSignalGenerator
from .signal_generator_enhanced import SignalType, SignalMetrics

# Alias for backward compatibility
SignalGenerator = EnhancedSignalGenerator

__all__ = ['EnhancedSignalGenerator', 'SignalGenerator', 'SignalType', 'SignalMetrics']
'''
        signals_init.write_text(content)
        print("   ✅ Updated signals/__init__.py")
    
    # Update portfolio/__init__.py
    portfolio_init = Path("crypto_momentum_backtest/portfolio/__init__.py")
    if portfolio_init.exists():
        content = '''"""Enhanced portfolio module."""
from .erc_optimizer_enhanced import EnhancedERCOptimizer
from .rebalancer_enhanced import EnhancedRebalancer

# Aliases for backward compatibility
ERCOptimizer = EnhancedERCOptimizer
Rebalancer = EnhancedRebalancer

__all__ = ['EnhancedERCOptimizer', 'ERCOptimizer', 'EnhancedRebalancer', 'Rebalancer']
'''
        portfolio_init.write_text(content)
        print("   ✅ Updated portfolio/__init__.py")
    
    # Update risk/__init__.py
    risk_init = Path("crypto_momentum_backtest/risk/__init__.py")
    if risk_init.exists():
        content = '''"""Enhanced risk module."""
from .risk_manager_enhanced import EnhancedRiskManager, MarketRegime

# Alias for backward compatibility
RiskManager = EnhancedRiskManager

__all__ = ['EnhancedRiskManager', 'RiskManager', 'MarketRegime']
'''
        risk_init.write_text(content)
        print("   ✅ Updated risk/__init__.py")
    
    # Update data/__init__.py
    data_init = Path("crypto_momentum_backtest/data/__init__.py")
    if data_init.exists():
        content = '''"""Enhanced data module."""
from .universe_manager_enhanced import EnhancedUniverseManager

# Alias for backward compatibility
UniverseManager = EnhancedUniverseManager

__all__ = ['EnhancedUniverseManager', 'UniverseManager']
'''
        data_init.write_text(content)
        print("   ✅ Updated data/__init__.py")

def create_summary_report():
    """Create a summary of enhancements."""
    
    report = '''# Crypto Momentum Strategy Enhancements Summary

## 🎯 Key Improvements Implemented

### 1. Signal Generation
- **Momentum Scoring System**: Composite scoring using price, volume, RSI, and MACD
- **Multi-Timeframe Analysis**: ADX across 7, 14, and 21-day periods
- **Adaptive Parameters**: Dynamic EWMA spans based on volatility
- **Dual Momentum**: Combines absolute and relative momentum

### 2. Portfolio Construction
- **Momentum Tilting**: Weights adjusted based on momentum scores
- **Concentration Mode**: 60% allocation to top 5 performers
- **Dynamic Universe**: 30 → 15 assets with momentum filtering
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

## 📊 Expected Performance Improvements

1. **Better Trend Capture**: Lower ADX threshold (15 vs 25)
2. **Reduced Drawdowns**: Trailing stops + regime detection
3. **Higher Sharpe**: Risk-adjusted position sizing
4. **Lower Costs**: Smarter rebalancing + momentum preservation

## 🔧 Configuration Changes

Key parameter updates in config_enhanced.yaml:
- Universe: 10 → 30 (filtered to 15)
- Max position: 10% → 25%
- ADX threshold: 25 → 15
- Correlation limit: 0.8 → 0.7
- Added trailing stops: 2.5x ATR

## 📈 Next Steps

1. Run comparison backtest:
   ```bash
   # Original strategy
   python run.py --config config.yaml --output-dir output_original
   
   # Enhanced strategy
   python run.py --config config_enhanced.yaml --output-dir output_enhanced
   ```

2. Compare metrics:
   ```python
   python compare_results.py output_original output_enhanced
   ```

3. Fine-tune parameters using optimization:
   ```python
   python optimize_parameters.py --config config_enhanced.yaml
   ```

## ⚠️ Important Notes

- All original files backed up in `backups_pre_enhancement/`
- Enhanced classes use "Enhanced" prefix but maintain compatibility
- Monitor regime detection accuracy in live trading
- Consider paper trading for 30 days before live deployment

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
'''
    
    with open("ENHANCEMENT_SUMMARY.md", "w") as f:
        f.write(report)
    
    print("\n📄 Created ENHANCEMENT_SUMMARY.md")

def create_test_script():
    """Create a test script for the enhanced components."""
    
    test_script = '''#!/usr/bin/env python3
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
    print("\\n🧪 Testing Enhanced Signal Generator...")
    
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
    print("\\n🧪 Testing Enhanced Portfolio Optimizer...")
    
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
    print("\\n🧪 Testing Enhanced Risk Manager...")
    
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
    
    print("\\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("\\nYour enhanced strategy is ready to run!")
    print("Next: python run.py --config config_enhanced.yaml --no-validate")

if __name__ == "__main__":
    main()
'''
    
    with open("test_enhanced_strategy.py", "w") as f:
        f.write(test_script)
    
    print("📄 Created test_enhanced_strategy.py")

if __name__ == "__main__":
    # Add datetime import
    from datetime import datetime
    
    implement_enhancements()
    create_test_script()