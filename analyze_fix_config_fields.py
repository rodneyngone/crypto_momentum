#!/usr/bin/env python3
"""
Analyze RiskConfig fields and fix configuration mismatches.
"""

from pathlib import Path
import yaml
import re


def analyze_risk_config_fields():
    """Extract the actual fields that RiskConfig accepts."""
    
    config_file = Path("crypto_momentum_backtest/utils/config.py")
    
    if not config_file.exists():
        print("❌ Could not find config.py")
        return []
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Find the RiskConfig dataclass definition
    risk_config_start = content.find("@dataclass\nclass RiskConfig:")
    if risk_config_start == -1:
        risk_config_start = content.find("class RiskConfig:")
    
    if risk_config_start == -1:
        print("❌ Could not find RiskConfig class")
        return []
    
    # Find the next class or end of RiskConfig
    next_class = content.find("\n@dataclass\nclass", risk_config_start + 1)
    if next_class == -1:
        next_class = content.find("\nclass", risk_config_start + 100)
    
    if next_class == -1:
        risk_config_content = content[risk_config_start:]
    else:
        risk_config_content = content[risk_config_start:next_class]
    
    # Extract field definitions
    # Look for patterns like "field_name: type = default"
    field_pattern = r'^\s+(\w+):\s+[\w\[\], \.]+(?:\s*=\s*.+)?$'
    
    fields = []
    for line in risk_config_content.split('\n'):
        match = re.match(field_pattern, line)
        if match and not line.strip().startswith('#'):
            field_name = match.group(1)
            if field_name not in ['self', 'cls'] and not field_name.startswith('_'):
                fields.append(field_name)
    
    print(f"✅ Found RiskConfig fields: {fields}")
    return fields


def fix_config_yaml(risk_fields):
    """Fix config.yaml to only include valid RiskConfig fields."""
    
    config_file = Path("config.yaml")
    
    if not config_file.exists():
        print("❌ config.yaml not found")
        return False
    
    # Load current config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'risk' not in config:
        print("⚠️  No risk section in config")
        return False
    
    # Map common mismatched fields to correct ones
    field_mapping = {
        'max_correlation': 'correlation_threshold',
        'max_drawdown_threshold': 'max_drawdown',
        'max_exchange_exposure': None,  # Remove if not in RiskConfig
        'volatility_regime_multiplier': 'volatility_multiple',
        'atr_period': None,  # Remove
        'atr_multiplier': None,  # Remove
    }
    
    # Fix risk section
    risk_config = config['risk'].copy()
    fixed_risk = {}
    removed_fields = []
    
    for key, value in risk_config.items():
        if key in risk_fields:
            # Field is valid
            fixed_risk[key] = value
        elif key in field_mapping:
            # Field needs to be mapped
            mapped_key = field_mapping[key]
            if mapped_key and mapped_key in risk_fields:
                fixed_risk[mapped_key] = value
                print(f"  Mapped {key} -> {mapped_key}")
            else:
                removed_fields.append(key)
        else:
            # Unknown field, remove it
            removed_fields.append(key)
    
    # Add default values for missing required fields
    risk_defaults = {
        'max_drawdown': 0.25,
        'daily_var_limit': 0.05,
        'position_var_limit': 0.10,
        'correlation_threshold': 0.80,
        'volatility_lookback': 20,
        'volatility_multiple': 2.5,
        'use_stop_losses': True,
        'stop_loss_pct': 0.08,
        'use_take_profits': False,
        'take_profit_pct': 0.15,
        'leverage_limit': 1.0,
        'sector_concentration_limit': 0.30,
        'risk_monitoring_frequency': 'daily'
    }
    
    for field, default_value in risk_defaults.items():
        if field in risk_fields and field not in fixed_risk:
            fixed_risk[field] = default_value
            print(f"  Added default for {field}: {default_value}")
    
    # Update config
    config['risk'] = fixed_risk
    
    # Ensure strategy section exists
    if 'strategy' not in config:
        config['strategy'] = {
            'universe_size': config.get('data', {}).get('universe_size', 10),
            'rebalance_frequency': config.get('portfolio', {}).get('rebalance_frequency', 'weekly'),
            'max_position_size': config.get('portfolio', {}).get('max_position_size', 0.10),
            'long_short': True
        }
    
    # Save fixed config
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Fixed risk config - removed: {removed_fields}")
    return True


def create_compatible_config():
    """Create a fully compatible configuration based on actual dataclass fields."""
    
    print("\n📝 Creating fully compatible configuration...")
    
    config = {
        'data': {
            'start_date': '2022-01-01',
            'end_date': '2023-12-31',
            'universe_size': 10,
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
            'momentum_threshold': 0.01,
            'momentum_ewma_span': 20,
            'mean_return_ewm_threshold': 0.003,
            'mean_return_ewm_span': 5,
            'mean_return_simple_threshold': 0.005,
            'mean_return_simple_window': 3,
            'adx_period': 14,
            'adx_threshold': 25.0,
            'rsi_period': 14,
            'rsi_oversold': 25.0,
            'rsi_overbought': 75.0,
            'use_volume_confirmation': True,
            'volume_threshold': 1.5,
            'volume_lookback': 20,
            'volume_filter_multiple': 1.5,
            'min_signal_gap': 2,
            'max_signals_per_period': 50,
            'signal_decay_periods': 5,
            'use_regime_detection': False,
            'volatility_adjustment': True,
            'correlation_filter': True,
            'correlation_threshold': 0.8,
            # Legacy fields
            'ewma_fast': 10,
            'ewma_slow': 30,
            'threshold': 0.01,
            'lookback': 20
        },
        'portfolio': {
            'initial_capital': 1000000.0,
            'max_position_size': 0.10,
            'min_position_size': 0.02,
            'rebalance_frequency': 'weekly',
            'rebalance_threshold': 0.10,
            'optimization_method': 'risk_parity',
            'target_volatility': 0.20,
            'max_concentration': 0.15,
            'sector_max_weight': 0.40,
            'consider_transaction_costs': True,
            'min_trade_size': 500.0,
            'target_cash_allocation': 0.0,
            'max_cash_allocation': 0.05
        },
        'risk': {
            # Only include fields that RiskConfig actually accepts
            'max_drawdown': 0.25,
            'daily_var_limit': 0.05,
            'position_var_limit': 0.10,
            'correlation_threshold': 0.80,
            'volatility_lookback': 20,
            'volatility_multiple': 2.5,
            'use_stop_losses': True,
            'stop_loss_pct': 0.08,
            'use_take_profits': False,
            'take_profit_pct': 0.15,
            'leverage_limit': 1.0,
            'sector_concentration_limit': 0.30,
            'risk_monitoring_frequency': 'daily',
            'alert_thresholds': {
                'drawdown_warning': 0.15,
                'volatility_warning': 0.40,
                'correlation_warning': 0.70
            }
        },
        'costs': {
            'maker_fee': 0.001,
            'taker_fee': 0.001,
            'base_spread': 0.0005,
            'impact_coefficient': 0.0002,
            'use_slippage': True,
            'slippage_coefficient': 0.0003,
            'max_slippage': 0.01,
            'funding_rate': 0.0001,
            'borrow_rate': 0.0002,
            'fill_probability': 0.95,
            'partial_fill_threshold': 0.50,
            'funding_lookback_days': 30
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
        'strategy': {
            'universe_size': 10,
            'rebalance_frequency': 'weekly',
            'max_position_size': 0.10,
            'long_short': True
        },
        'random_seed': 42,
        'log_level': 'INFO',
        'parallel_processing': True,
        'max_workers': 4
    }
    
    # Save as config_compatible.yaml
    with open('config_compatible.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("✅ Created config_compatible.yaml")
    return True


def store_engine_params_separately():
    """Create a separate config for BacktestEngine-specific parameters."""
    
    engine_params = {
        'engine_params': {
            'max_correlation': 0.80,
            'max_exchange_exposure': 0.40,
            'volatility_regime_multiplier': 1.5,
            'atr_period': 14,
            'atr_multiplier': 2.0
        }
    }
    
    with open('engine_params.yaml', 'w') as f:
        yaml.dump(engine_params, f, default_flow_style=False)
    
    print("✅ Created engine_params.yaml for BacktestEngine-specific parameters")
    return True


def main():
    """Main fix function."""
    print("🔧 Analyzing and Fixing Configuration Field Mismatches")
    print("=" * 60)
    
    # Step 1: Analyze what fields RiskConfig actually accepts
    print("\n1️⃣ Analyzing RiskConfig fields...")
    risk_fields = analyze_risk_config_fields()
    
    if not risk_fields:
        # Fallback to known fields
        risk_fields = [
            'max_drawdown', 'daily_var_limit', 'position_var_limit',
            'correlation_threshold', 'volatility_lookback', 'volatility_multiple',
            'use_stop_losses', 'stop_loss_pct', 'use_take_profits', 'take_profit_pct',
            'leverage_limit', 'sector_concentration_limit', 'risk_monitoring_frequency',
            'alert_thresholds'
        ]
        print("⚠️  Using fallback field list")
    
    # Step 2: Fix current config.yaml
    print("\n2️⃣ Fixing config.yaml...")
    fix_config_yaml(risk_fields)
    
    # Step 3: Create compatible config
    create_compatible_config()
    
    # Step 4: Store engine params separately
    print("\n3️⃣ Creating separate engine parameters...")
    store_engine_params_separately()
    
    print("\n✅ Configuration fixes complete!")
    print("\n📋 Next steps:")
    print("1. Try running: python run.py --no-validate")
    print("2. If still having issues:")
    print("   copy config_compatible.yaml config.yaml")
    print("   python run.py --no-validate")
    print("\n💡 The BacktestEngine may need to be updated to:")
    print("- Use config.risk.correlation_threshold instead of max_correlation")
    print("- Handle missing fields with defaults")
    print("- Get ATR parameters from signals config instead of risk config")


if __name__ == "__main__":
    main()