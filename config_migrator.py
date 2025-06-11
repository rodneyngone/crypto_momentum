#!/usr/bin/env python3
"""
Configuration migration script for crypto momentum backtesting framework.
Converts legacy config files to new format with crypto-optimized defaults.
"""

import yaml
import shutil
from pathlib import Path
from datetime import datetime

def backup_config(config_path="config.yaml"):
    """Create a backup of the existing config file."""
    config_file = Path(config_path)
    if config_file.exists():
        backup_path = config_file.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml')
        shutil.copy(config_file, backup_path)
        print(f"✅ Created backup: {backup_path}")
        return backup_path
    return None

def migrate_config(config_path="config.yaml"):
    """Migrate legacy config to new format."""
    
    print("🔧 CRYPTO MOMENTUM CONFIG MIGRATOR")
    print("="*50)
    
    # Backup existing config
    backup_path = backup_config(config_path)
    
    # Load existing config
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"❌ Config file {config_path} not found")
        return
    
    try:
        with open(config_file, 'r') as f:
            old_config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return
    
    if old_config is None:
        old_config = {}
    
    print(f"📁 Loaded existing config from {config_path}")
    
    # Create new config structure with crypto-optimized defaults
    new_config = {
        'data': {
            'start_date': old_config.get('data', {}).get('start_date', '2022-01-01'),
            'end_date': old_config.get('data', {}).get('end_date', '2023-12-31'),
            'universe_size': old_config.get('data', {}).get('universe_size', 10),
            'market_cap_threshold': 100000000,
            'exclude_stablecoins': True,
            'survivorship_bias': True,
            'data_source': 'binance',
            'cache_directory': 'data',
            'cache_days': 30,
            'update_frequency': 'daily',
            'min_trading_days': 100,
            'max_missing_data_pct': 0.05
        },
        'signals': {
            'signal_strategy': 'momentum',
            # CRYPTO-OPTIMIZED THRESHOLDS (based on empirical analysis)
            'momentum_threshold': 0.01,                    # 1% for crypto momentum
            'momentum_ewma_span': 20,
            'mean_return_ewm_threshold': 0.015,            # 1.5% - FIXED for crypto
            'mean_return_ewm_span': 10,                    # More responsive for crypto
            'mean_return_simple_threshold': 0.015,         # 1.5% - FIXED for crypto
            'mean_return_simple_window': 10,               # Balanced for crypto
            'adx_period': 14,
            'adx_threshold': 20.0,
            'rsi_period': 14,
            'rsi_oversold': 30.0,
            'rsi_overbought': 70.0,
            'use_volume_confirmation': True,
            'volume_threshold': 1.2,
            'volume_lookback': 20,
            'min_signal_gap': 1,
            'max_signals_per_period': 100,
            'signal_decay_periods': 5,
            'use_regime_detection': False,
            'volatility_adjustment': True,
            'correlation_filter': False,
            'correlation_threshold': 0.8
        },
        'portfolio': {
            'initial_capital': old_config.get('portfolio', {}).get('initial_capital', 1000000.0),
            'max_position_size': old_config.get('portfolio', {}).get('max_position_size', 0.10),
            'min_position_size': 0.01,
            'rebalance_frequency': old_config.get('portfolio', {}).get('rebalance_frequency', 'monthly'),
            'rebalance_threshold': 0.05,
            'optimization_method': 'risk_parity',
            'target_volatility': 0.15,
            'max_concentration': 0.20,
            'sector_max_weight': 0.30,
            'consider_transaction_costs': True,
            'min_trade_size': 100.0,
            'target_cash_allocation': 0.0,
            'max_cash_allocation': 0.05
        },
        'risk': {
            'max_drawdown': old_config.get('risk', {}).get('max_drawdown', 0.20),
            'daily_var_limit': 0.05,
            'position_var_limit': 0.10,
            'correlation_threshold': 0.80,
            'volatility_lookback': 30,
            'volatility_multiple': 2.0,
            'use_stop_losses': True,
            'stop_loss_pct': 0.05,
            'use_take_profits': False,
            'take_profit_pct': 0.10,
            'leverage_limit': 1.0,
            'sector_concentration_limit': 0.30,
            'risk_monitoring_frequency': 'daily',
            'alert_thresholds': {
                'drawdown_warning': 0.10,
                'volatility_warning': 0.25,
                'correlation_warning': 0.70
            }
        },
        'costs': {
            'maker_fee': 0.001,
            'taker_fee': 0.001,
            'base_spread': 0.0005,
            'impact_coefficient': 0.0001,
            'use_slippage': True,
            'slippage_coefficient': 0.0002,
            'max_slippage': 0.005,
            'funding_rate': 0.0001,
            'borrow_rate': 0.0002,
            'fill_probability': 0.95,
            'partial_fill_threshold': 0.50
        },
        'backtest': {
            'execution_delay': 0,
            'market_hours_only': False,
            'benchmark_symbol': 'BTCUSDT',
            'risk_free_rate': 0.02,
            'walk_forward_analysis': False,
            'out_of_sample_periods': 252,
            'monte_carlo_runs': 1000,
            'save_trades': True,
            'save_positions': True,
            'save_metrics': True,
            'output_directory': 'output',
            'calculate_attribution': True,
            'attribution_frequency': 'monthly'
        },
        'random_seed': 42,
        'log_level': 'INFO',
        'parallel_processing': True,
        'max_workers': 4
    }
    
    # Preserve any existing values from old config
    print("🔄 Migrating existing parameters...")
    migrated_params = []
    
    # Migrate data section
    if 'data' in old_config:
        for key, value in old_config['data'].items():
            if key in new_config['data']:
                new_config['data'][key] = value
                migrated_params.append(f"data.{key}")
    
    # Migrate portfolio section
    if 'portfolio' in old_config:
        for key, value in old_config['portfolio'].items():
            if key in new_config['portfolio']:
                new_config['portfolio'][key] = value
                migrated_params.append(f"portfolio.{key}")
    
    # Migrate risk section
    if 'risk' in old_config:
        for key, value in old_config['risk'].items():
            if key in new_config['risk']:
                new_config['risk'][key] = value
                migrated_params.append(f"risk.{key}")
    
    # Handle legacy signal parameters
    legacy_signal_mapping = {
        'threshold': 'momentum_threshold',
        'ewma_fast': 'momentum_ewma_span',
        'combine_signals': None,  # Remove this parameter
        'signal_type': 'signal_strategy'
    }
    
    if 'signals' in old_config:
        legacy_params = []
        for old_key, value in old_config['signals'].items():
            if old_key in legacy_signal_mapping:
                new_key = legacy_signal_mapping[old_key]
                if new_key:  # Only migrate if not None (i.e., not removed)
                    new_config['signals'][new_key] = value
                    migrated_params.append(f"signals.{old_key} -> signals.{new_key}")
                else:
                    legacy_params.append(old_key)
            elif old_key in new_config['signals']:
                new_config['signals'][old_key] = value
                migrated_params.append(f"signals.{old_key}")
            else:
                legacy_params.append(old_key)
        
        if legacy_params:
            print(f"⚠️  Removed legacy parameters: {', '.join(legacy_params)}")
    
    # Save new config
    try:
        with open(config_file, 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False, sort_keys=False, indent=2)
        
        print(f"✅ Migrated config saved to {config_path}")
        
        if migrated_params:
            print(f"📋 Migrated {len(migrated_params)} parameters:")
            for param in migrated_params[:10]:  # Show first 10
                print(f"   {param}")
            if len(migrated_params) > 10:
                print(f"   ... and {len(migrated_params) - 10} more")
        
        print(f"\n🎯 KEY CHANGES MADE:")
        print(f"   ✅ Mean return EWM threshold: 0.015 (crypto-optimized)")
        print(f"   ✅ Mean return simple threshold: 0.015 (crypto-optimized)")
        print(f"   ✅ EWM span: 10 (more responsive for crypto)")
        print(f"   ✅ Simple window: 10 (balanced for crypto)")
        print(f"   ✅ Removed legacy 'combine_signals' parameter")
        print(f"   ✅ Added comprehensive risk and cost configurations")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Test the migrated config:")
        print(f"      python run.py --no-validate")
        print(f"   2. Test mean return signals:")
        print(f"      python test_mean_return_signals_FIXED.py")
        print(f"   3. If needed, restore backup from: {backup_path}")
        
    except Exception as e:
        print(f"❌ Error saving migrated config: {e}")
        if backup_path:
            print(f"💡 Restore from backup: copy {backup_path} config.yaml")

def main():
    """Main migration function."""
    migrate_config()

if __name__ == "__main__":
    main()