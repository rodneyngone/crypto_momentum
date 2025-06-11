# File: verify_enhancements.py (NEW)
"""Verify all enhancements are properly integrated."""

import sys
from pathlib import Path

def verify_enhancements():
    """Check that all enhanced components are in place."""
    
    checks = {
        'Signal Generator': {
            'file': 'crypto_momentum_backtest/signals/signal_generator.py',
            'enhanced_markers': ['EnhancedSignalGenerator', 'SignalType', 'momentum_weights']
        },
        'Portfolio Optimizer': {
            'file': 'crypto_momentum_backtest/portfolio/erc_optimizer.py',
            'enhanced_markers': ['EnhancedERCOptimizer', 'concentration_mode', 'momentum_tilt_strength']
        },
        'Risk Manager': {
            'file': 'crypto_momentum_backtest/risk/risk_manager.py',
            'enhanced_markers': ['EnhancedRiskManager', 'MarketRegime', 'detect_market_regime']
        },
        'Rebalancer': {
            'file': 'crypto_momentum_backtest/portfolio/rebalancer.py',
            'enhanced_markers': ['EnhancedRebalancer', 'regime_frequency_map', 'momentum_preservation']
        },
        'Universe Manager': {
            'file': 'crypto_momentum_backtest/data/universe_manager.py',
            'enhanced_markers': ['EnhancedUniverseManager', 'momentum_filter', 'category_limits']
        }
    }
    
    print("🔍 Verifying Enhanced Components")
    print("=" * 50)
    
    all_verified = True
    
    for component, info in checks.items():
        filepath = Path(info['file'])
        if filepath.exists():
            content = filepath.read_text(encoding='utf-8')
            markers_found = all(marker in content for marker in info['enhanced_markers'])
            
            if markers_found:
                print(f"✅ {component}: Enhanced version active")
            else:
                print(f"❌ {component}: Missing enhanced markers")
                all_verified = False
        else:
            print(f"❌ {component}: File not found")
            all_verified = False
    
    print("=" * 50)
    
    if all_verified:
        print("✅ All enhancements verified!")
        
        # Check config
        config_path = Path('config.yaml')
        if config_path.exists():
            config_content = config_path.read_text(encoding='utf-8')
            if 'adx_periods' in config_content and 'concentration_mode' in config_content:
                print("✅ Enhanced configuration active")
            else:
                print("⚠️  Using standard configuration")
    else:
        print("❌ Some enhancements missing - check above")
    
    return all_verified

if __name__ == "__main__":
    verify_enhancements()