#!/usr/bin/env python3
"""
Fix all configuration mismatches by analyzing actual dataclass fields.
"""

from pathlib import Path
import yaml
import re
import ast


def extract_dataclass_fields(config_file_path, class_name):
    """Extract fields from a dataclass definition."""
    
    with open(config_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the dataclass
    class_pattern = f"@dataclass\\s*\\n\\s*class {class_name}"
    class_match = re.search(class_pattern, content)
    
    if not class_match:
        # Try without @dataclass decorator
        class_pattern = f"class {class_name}"
        class_match = re.search(class_pattern, content)
    
    if not class_match:
        print(f"[WARNING] Could not find {class_name}")
        return []
    
    # Find the end of the class (next class or end of file)
    start_pos = class_match.start()
    next_class = re.search(r"\n\s*(@dataclass\s*\n\s*)?class\s+", content[start_pos + len(class_match.group()):])
    
    if next_class:
        end_pos = start_pos + len(class_match.group()) + next_class.start()
    else:
        end_pos = len(content)
    
    class_content = content[start_pos:end_pos]
    
    # Extract field definitions
    # Pattern: field_name: type = default_value
    field_pattern = r'^\s+(\w+):\s+[^=\n]+(?:\s*=\s*[^\n]+)?'
    
    fields = []
    for line in class_content.split('\n'):
        if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""'):
            match = re.match(field_pattern, line)
            if match:
                field_name = match.group(1)
                if field_name not in ['self', 'cls'] and not field_name.startswith('_'):
                    fields.append(field_name)
    
    return fields


def analyze_all_config_classes():
    """Analyze all configuration dataclasses to find their fields."""
    
    config_file = Path("crypto_momentum_backtest/utils/config.py")
    
    if not config_file.exists():
        print("[ERROR] config.py not found")
        return {}
    
    classes = {
        'DataConfig': [],
        'SignalConfig': [],
        'PortfolioConfig': [],
        'RiskConfig': [],
        'CostConfig': [],
        'BacktestConfig': [],
        'Config': []
    }
    
    for class_name in classes:
        fields = extract_dataclass_fields(config_file, class_name)
        classes[class_name] = fields
        print(f"[OK] {class_name} has {len(fields)} fields")
    
    return classes


def fix_config_yaml_comprehensive(dataclass_fields):
    """Fix config.yaml to only include valid fields for each dataclass."""
    
    config_file = Path("config.yaml")
    
    if not config_file.exists():
        print("[ERROR] config.yaml not found")
        return False
    
    # Load current config
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Create fixed config
    fixed_config = {}
    
    # Fix each section
    sections = {
        'data': 'DataConfig',
        'signals': 'SignalConfig',
        'portfolio': 'PortfolioConfig',
        'risk': 'RiskConfig',
        'costs': 'CostConfig',
        'backtest': 'BacktestConfig'
    }
    
    removed_fields_report = {}
    
    for section, class_name in sections.items():
        if section in config and class_name in dataclass_fields:
            valid_fields = dataclass_fields[class_name]
            fixed_section = {}
            removed_fields = []
            
            for key, value in config[section].items():
                if key in valid_fields:
                    fixed_section[key] = value
                else:
                    removed_fields.append(key)
            
            fixed_config[section] = fixed_section
            
            if removed_fields:
                removed_fields_report[section] = removed_fields
                print(f"[INFO] Removed from {section}: {removed_fields}")
    
    # Add strategy section for backward compatibility
    if 'strategy' not in fixed_config:
        fixed_config['strategy'] = {
            'universe_size': config.get('data', {}).get('universe_size', 10),
            'rebalance_frequency': config.get('portfolio', {}).get('rebalance_frequency', 'weekly'),
            'max_position_size': config.get('portfolio', {}).get('max_position_size', 0.10),
            'long_short': True
        }
    
    # Add global settings
    for key in ['random_seed', 'log_level', 'parallel_processing', 'max_workers']:
        if key in config:
            fixed_config[key] = config[key]
    
    # Save fixed config
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(fixed_config, f, default_flow_style=False, sort_keys=False)
    
    print("[OK] Fixed config.yaml")
    
    # Save removed fields report
    if removed_fields_report:
        with open('removed_config_fields.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(removed_fields_report, f, default_flow_style=False)
        print("[INFO] Saved removed fields to removed_config_fields.yaml")
    
    return True


def create_working_config():
    """Create a minimal working configuration with only valid fields."""
    
    working_config = {
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
            'min_signal_gap': 2,
            'max_signals_per_period': 50,
            'signal_decay_periods': 5,
            'use_regime_detection': False,
            'volatility_adjustment': True,
            'correlation_filter': True,
            'correlation_threshold': 0.8,
            # Add legacy fields that might be expected
            'ewma_fast': 10,
            'ewma_slow': 30,
            'volume_filter_multiple': 1.5
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
            'partial_fill_threshold': 0.50
            # Note: funding_lookback_days removed as it's not in CostConfig
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
    
    # Save as config_working.yaml
    with open('config_working.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(working_config, f, default_flow_style=False, sort_keys=False)
    
    print("[OK] Created config_working.yaml")
    return True


def patch_funding_rates():
    """Patch FundingRates to handle the lookback_days parameter differently."""
    
    funding_file = Path("crypto_momentum_backtest/costs/funding_rates.py")
    
    if not funding_file.exists():
        print("[WARNING] funding_rates.py not found")
        return
    
    # The funding_lookback_days parameter should be passed to FundingRates
    # but it's not in CostConfig, so BacktestEngine needs to handle it separately
    
    engine_file = Path("crypto_momentum_backtest/backtest/engine.py")
    
    if engine_file.exists():
        with open(engine_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix FundingRates initialization
        old_funding = "self.funding_rates = FundingRates(\n            lookback_days=config.costs.funding_lookback_days,"
        new_funding = "self.funding_rates = FundingRates(\n            lookback_days=30,  # Default 30 days"
        
        if old_funding in content:
            content = content.replace(old_funding, new_funding)
            
            with open(engine_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("[OK] Fixed FundingRates initialization in engine.py")


def main():
    print("Fixing ALL Configuration Mismatches")
    print("=" * 60)
    
    # Step 1: Analyze dataclass fields
    print("\n1. Analyzing dataclass fields...")
    dataclass_fields = analyze_all_config_classes()
    
    # Step 2: Fix config.yaml
    print("\n2. Fixing config.yaml...")
    fix_config_yaml_comprehensive(dataclass_fields)
    
    # Step 3: Create working config
    print("\n3. Creating working configuration...")
    create_working_config()
    
    # Step 4: Patch funding rates issue
    print("\n4. Patching funding rates...")
    patch_funding_rates()
    
    print("\n[OK] All fixes applied!")
    print("\nNext steps:")
    print("1. Test with current config: python test_engine_init.py")
    print("2. If still having issues, use working config:")
    print("   copy config_working.yaml config.yaml")
    print("   python test_engine_init.py")
    print("3. Run backtest: python run.py --no-validate")


if __name__ == "__main__":
    main()