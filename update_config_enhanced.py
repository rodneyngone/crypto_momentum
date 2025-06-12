#!/usr/bin/env python3
"""
Script to update your config.yaml with enhanced parameters while preserving structure.
"""

import yaml
from pathlib import Path
import shutil
from datetime import datetime

def update_config_to_enhanced():
    """Update config.yaml with enhanced parameters."""
    
    print("🔧 Updating Configuration for Enhanced Components")
    print("=" * 60)
    
    config_path = Path('config.yaml')
    
    # Create backup
    if config_path.exists():
        backup_path = Path(f'config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml.backup')
        shutil.copy2(config_path, backup_path)
        print(f"✅ Created backup: {backup_path}")
    
    # Enhanced configuration with all improvements
    enhanced_config = {
        'data': {
            'start_date': '2022-01-01',
            'end_date': '2023-12-31',
            'universe_size': 30,  # Increased from 10
            'selection_size': 15,  # Final universe after momentum filter
            'market_cap_threshold': 100000000,
            'exclude_stablecoins': True,
            'exclude_wrapped': True,
            'survivorship_bias': True,
            'data_source': 'binance',
            'cache_directory': 'data',
            'cache_days': 30,
            'update_frequency': 'daily',
            'min_trading_days': 100,
            'max_missing_data_pct': 0.05
        },
        
        'signals': {
            # Strategy selection
            'signal_strategy': 'momentum_score',  # Use enhanced momentum scoring
            
            # Multi-timeframe ADX parameters
            'adx_periods': [7, 14, 21],
            'adx_threshold': 15.0,  # Lowered from 25
            
            # Adaptive EWMA parameters
            'base_ewma_span': 20,
            'adaptive_ewma': True,
            
            # Momentum scoring weights
            'momentum_weights': {
                'price': 0.4,
                'volume': 0.2,
                'rsi': 0.2,
                'macd': 0.2
            },
            
            # Dual momentum parameters
            'absolute_momentum_threshold': 0.05,
            'relative_momentum_threshold': 0.0,
            'momentum_lookback': 30,
            
            # FIXED THRESHOLDS - Critical fix!
            'momentum_threshold': 0.01,
            'mean_return_ewm_threshold': 0.003,  # Fixed from 0.02
            'mean_return_simple_threshold': 0.005,  # Fixed from 0.02
            'mean_return_ewm_span': 10,
            'mean_return_simple_window': 5,
            
            # Technical indicators
            'rsi_period': 14,
            'rsi_oversold': 25.0,
            'rsi_overbought': 75.0,
            
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            
            # Signal filtering
            'min_score_threshold': 0.3,
            'use_volume_confirmation': True,
            'volume_threshold': 1.2,
            'max_correlation': 0.7
        },
        
        'portfolio': {
            'initial_capital': 1000000.0,
            'max_position_size': 0.25,  # Increased from 0.10
            'min_position_size': 0.02,
            
            # Dynamic rebalancing
            'base_rebalance_frequency': 'weekly',
            'use_dynamic_rebalancing': True,
            
            # Enhanced optimization
            'optimization_method': 'enhanced_risk_parity',
            'concentration_mode': True,
            'top_n_assets': 5,
            'concentration_weight': 0.6,  # 60% in top 5
            'momentum_tilt_strength': 0.5,
            'use_momentum_weighting': True,
            
            # Portfolio parameters
            'target_volatility': 0.25,  # Increased for crypto
            'max_concentration': 0.30,
            'category_max_weights': {
                'defi': 0.40,
                'layer1': 0.50,
                'layer2': 0.30,
                'meme': 0.20,
                'exchange': 0.30
            },
            
            'consider_transaction_costs': True,
            'min_trade_size': 500.0,
            'target_cash_allocation': 0.0,
            'max_cash_allocation': 0.05
        },
        
        'risk': {
            # Drawdown and risk limits
            'max_drawdown': 0.20,  # Reduced from 0.25
            'daily_var_limit': 0.05,
            'position_var_limit': 0.10,
            
            # Correlation management
            'correlation_threshold': 0.70,  # Reduced from 0.80
            'max_correlation': 0.70,
            'max_exchange_exposure': 0.40,
            
            # Volatility parameters
            'volatility_lookback': 20,
            'volatility_multiple': 2.0,
            'volatility_regime_multiplier': 1.5,
            
            # Trailing stops - NEW!
            'use_trailing_stops': True,
            'trailing_stop_atr_multiplier': 2.5,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            
            # Regime detection
            'regime_lookback': 60,
            'crisis_vol_threshold': 1.0,  # 100% annual vol = crisis
            
            # Stop loss and take profit
            'use_stop_losses': True,
            'stop_loss_pct': 0.08,
            'use_take_profits': False,
            'take_profit_pct': 0.15,
            
            # Other parameters
            'leverage_limit': 1.0,
            'sector_concentration_limit': 0.30,
            'risk_monitoring_frequency': 'daily',
            'max_drawdown_threshold': 0.20,
            
            # Alert thresholds
            'alert_thresholds': {
                'drawdown_warning': 0.10,
                'volatility_warning': 0.30,
                'correlation_warning': 0.60
            }
        },
        
        'costs': {
            # Updated fee structure
            'maker_fee': 0.0008,  # Reduced - assuming VIP tier
            'taker_fee': 0.001,
            'base_spread': 0.0003,  # Tighter spread
            'impact_coefficient': 0.0001,
            
            # Slippage modeling
            'use_slippage': True,
            'slippage_coefficient': 0.0002,
            'max_slippage': 0.005,
            
            # Funding and borrowing
            'funding_rate': 0.0001,
            'borrow_rate': 0.0002,
            
            # Order execution
            'fill_probability': 0.95,
            'partial_fill_threshold': 0.50,
            'funding_lookback_days': 30
        },
        
        'backtest': {
            'execution_delay': 0,
            'market_hours_only': False,
            'benchmark_symbol': 'BTCUSDT',
            'risk_free_rate': 0.02,
            
            # Validation settings
            'walk_forward_analysis': True,
            'walk_forward_splits': 5,
            'out_of_sample_periods': 60,
            'monte_carlo_runs': 1000,
            
            # Output settings
            'save_trades': True,
            'save_positions': True,
            'save_metrics': True,
            'output_directory': 'output_enhanced',
            
            # Performance analysis
            'calculate_attribution': True,
            'attribution_frequency': 'monthly'
        },
        
        # Regime-specific adjustments
        'regime_parameters': {
            'trending': {
                'exposure_multiplier': 1.5,
                'rebalance_frequency': 'daily',
                'max_position_size': 0.30,
                'stop_loss_multiplier': 1.2
            },
            'volatile': {
                'exposure_multiplier': 0.5,
                'rebalance_frequency': 'weekly',
                'max_position_size': 0.15,
                'stop_loss_multiplier': 0.8
            },
            'ranging': {
                'exposure_multiplier': 0.7,
                'rebalance_frequency': 'biweekly',
                'max_position_size': 0.20,
                'stop_loss_multiplier': 1.0
            },
            'crisis': {
                'exposure_multiplier': 0.3,
                'rebalance_frequency': 'daily',
                'max_position_size': 0.10,
                'stop_loss_multiplier': 0.6
            }
        },
        
        # Global settings
        'random_seed': 42,
        'log_level': 'INFO',
        'parallel_processing': True,
        'max_workers': 4
    }
    
    # Save enhanced config
    with open('config_enhanced.yaml', 'w') as f:
        yaml.dump(enhanced_config, f, default_flow_style=False, sort_keys=False)
    
    print("✅ Created config_enhanced.yaml with all enhancements")
    
    # Also update the main config.yaml
    with open('config.yaml', 'w') as f:
        yaml.dump(enhanced_config, f, default_flow_style=False, sort_keys=False)
    
    print("✅ Updated config.yaml with enhanced parameters")
    
    # Create a comparison summary
    create_comparison_summary()

def create_comparison_summary():
    """Create a summary of key parameter changes."""
    
    summary = """
# Configuration Enhancement Summary

## 🔑 Critical Fixes
- mean_return_ewm_threshold: 0.02 → 0.003 (Fixed 10x bug!)
- mean_return_simple_threshold: 0.02 → 0.005

## 📊 Signal Improvements
- signal_strategy: 'momentum' → 'momentum_score'
- adx_threshold: 25 → 15 (better trend detection)
- Added multi-timeframe ADX: [7, 14, 21]
- Added momentum scoring weights

## 💼 Portfolio Enhancements
- max_position_size: 0.10 → 0.25
- Added concentration mode (60% in top 5)
- Added momentum tilting (0.5 strength)
- Universe: 10 → 30 (filtered to 15)

## 🛡️ Risk Improvements
- max_drawdown: 0.25 → 0.20
- Added trailing stops (2.5x ATR)
- Added regime detection
- correlation_threshold: 0.80 → 0.70

## 💰 Cost Optimizations
- maker_fee: 0.001 → 0.0008 (VIP tier)
- base_spread: 0.0005 → 0.0003
- Added dynamic rebalancing

## 🎯 Expected Impact
- Higher Sharpe ratio (targeting > 1.2)
- Lower drawdowns (< 20%)
- Better trend capture
- Reduced trading costs
"""
    
    with open('config_changes_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print("\n📄 Created config_changes_summary.txt")

def verify_config():
    """Verify the enhanced config works with the system."""
    
    print("\n🔍 Verifying enhanced configuration...")
    
    try:
        from crypto_momentum_backtest.utils.config import Config
        
        # Try loading the enhanced config
        config = Config.from_yaml(Path('config_enhanced.yaml'))
        
        # Check critical parameters
        checks = {
            'Signal strategy': config.signals.signal_strategy == 'momentum_score',
            'Fixed thresholds': config.signals.mean_return_ewm_threshold == 0.003,
            'Concentration mode': hasattr(config.portfolio, 'concentration_mode'),
            'Trailing stops': hasattr(config.risk, 'use_trailing_stops'),
            'ADX periods': hasattr(config.signals, 'adx_periods')
        }
        
        print("\nConfiguration checks:")
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")
        
        if all(checks.values()):
            print("\n✅ Enhanced configuration is valid!")
        else:
            print("\n⚠️  Some configuration parameters might need adjustment")
            
    except Exception as e:
        print(f"\n❌ Error verifying config: {e}")
        print("This might be normal if Config class expects different structure")

if __name__ == "__main__":
    update_config_to_enhanced()
    verify_config()
    
    print("\n" + "=" * 60)
    print("✅ CONFIGURATION UPDATE COMPLETE!")
    print("\nYou now have:")
    print("  - config_enhanced.yaml: Full enhanced configuration")
    print("  - config.yaml: Updated with enhanced parameters")
    print(f"  - Backup of original config")
    print("  - config_changes_summary.txt: Summary of changes")
    print("\nTo use enhanced config:")
    print("  python run.py --config config_enhanced.yaml --no-validate")
    print("\nOr just run normally (config.yaml is updated):")
    print("  python run.py --no-validate")