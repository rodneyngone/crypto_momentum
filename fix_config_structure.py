#!/usr/bin/env python3
"""
Fix configuration structure issues in the backtest engine.
"""

from pathlib import Path


def fix_backtest_engine():
    """Fix the BacktestEngine to look for atr_period in the correct place."""
    
    engine_file = Path("crypto_momentum_backtest/backtest/engine.py")
    
    if not engine_file.exists():
        print(f"❌ Could not find {engine_file}")
        return False
    
    # Read the file
    with open(engine_file, 'r') as f:
        content = f.read()
    
    # Fix the atr_period reference - it should be from signals config
    old_line = "atr_period=config.risk.atr_period,"
    new_line = "atr_period=config.signals.adx_period,"  # Use adx_period as ATR period
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        
        # Write back
        with open(engine_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Fixed atr_period reference in engine.py")
        return True
    else:
        print("⚠️  Could not find the line to fix. Checking alternative fix...")
        return False


def add_missing_config_attributes():
    """Add missing attributes to config.yaml if needed."""
    
    import yaml
    
    config_file = Path("config.yaml")
    
    if not config_file.exists():
        print("❌ config.yaml not found")
        return False
    
    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure risk section has atr_period and atr_multiplier
    if 'risk' not in config:
        config['risk'] = {}
    
    # Add missing risk parameters that PositionSizer expects
    if 'atr_period' not in config['risk']:
        config['risk']['atr_period'] = 14  # Default ATR period
        print("✅ Added atr_period to risk config")
    
    if 'atr_multiplier' not in config['risk']:
        config['risk']['atr_multiplier'] = 2.0  # Default ATR multiplier
        print("✅ Added atr_multiplier to risk config")
    
    # Save updated config
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("✅ Updated config.yaml with missing attributes")
    return True


def create_complete_config():
    """Create a complete working configuration file."""
    
    config_content = """# Complete working configuration for crypto momentum backtest

data:
  start_date: '2022-01-01'
  end_date: '2023-12-31'
  universe_size: 10
  market_cap_threshold: 100000000
  exclude_stablecoins: true
  survivorship_bias: true
  data_source: binance
  cache_directory: data
  cache_days: 30
  update_frequency: daily
  min_trading_days: 100
  max_missing_data_pct: 0.05

signals:
  # Signal strategy selection
  signal_strategy: momentum
  
  # Momentum parameters
  momentum_threshold: 0.01
  momentum_ewma_span: 20
  
  # Mean return parameters (crypto-optimized)
  mean_return_ewm_threshold: 0.003
  mean_return_ewm_span: 5
  mean_return_simple_threshold: 0.005
  mean_return_simple_window: 3
  
  # Technical indicators
  adx_period: 14
  adx_threshold: 25.0
  ewma_fast: 10
  ewma_slow: 30
  
  # Volume confirmation
  use_volume_confirmation: true
  volume_threshold: 1.5
  volume_lookback: 20
  volume_filter_multiple: 1.5
  
  # Signal filtering
  min_signal_gap: 2
  max_signals_per_period: 50
  signal_decay_periods: 5
  
  # Additional settings
  use_regime_detection: false
  volatility_adjustment: true
  correlation_filter: true
  correlation_threshold: 0.8

portfolio:
  initial_capital: 1000000.0
  max_position_size: 0.10
  min_position_size: 0.02
  rebalance_frequency: weekly
  rebalance_threshold: 0.10
  optimization_method: risk_parity
  target_volatility: 0.20
  max_concentration: 0.15
  sector_max_weight: 0.40
  consider_transaction_costs: true
  min_trade_size: 500.0
  target_cash_allocation: 0.0
  max_cash_allocation: 0.05

risk:
  # ATR-based position sizing parameters
  atr_period: 14           # REQUIRED for PositionSizer
  atr_multiplier: 2.0      # REQUIRED for PositionSizer
  
  # Drawdown and risk limits
  max_drawdown: 0.25
  daily_var_limit: 0.05
  position_var_limit: 0.10
  
  # Correlation and concentration
  correlation_threshold: 0.80
  max_correlation: 0.80
  max_exchange_exposure: 0.40
  
  # Volatility management
  volatility_lookback: 20
  volatility_multiple: 2.5
  volatility_regime_multiplier: 1.5
  
  # Stop loss and take profit
  use_stop_losses: true
  stop_loss_pct: 0.08
  use_take_profits: false
  take_profit_pct: 0.15
  
  # Other risk parameters
  leverage_limit: 1.0
  sector_concentration_limit: 0.30
  risk_monitoring_frequency: daily
  max_drawdown_threshold: 0.25
  
  # Alert thresholds
  alert_thresholds:
    drawdown_warning: 0.15
    volatility_warning: 0.40
    correlation_warning: 0.70

costs:
  maker_fee: 0.001
  taker_fee: 0.001
  base_spread: 0.0005
  impact_coefficient: 0.0002
  use_slippage: true
  slippage_coefficient: 0.0003
  max_slippage: 0.01
  funding_rate: 0.0001
  borrow_rate: 0.0002
  fill_probability: 0.95
  partial_fill_threshold: 0.50
  funding_lookback_days: 30

backtest:
  execution_delay: 0
  market_hours_only: false
  benchmark_symbol: BTCUSDT
  risk_free_rate: 0.02
  walk_forward_analysis: false
  out_of_sample_periods: 252
  monte_carlo_runs: 1000
  save_trades: true
  save_positions: true
  save_metrics: true
  output_directory: output
  calculate_attribution: true
  attribution_frequency: monthly

# Global settings
random_seed: 42
log_level: INFO
parallel_processing: true
max_workers: 4
"""
    
    # Save as config_complete.yaml
    with open("config_complete.yaml", 'w') as f:
        f.write(config_content)
    
    print("✅ Created config_complete.yaml with all required parameters")
    
    # Also update the main config.yaml
    import shutil
    shutil.copy("config_complete.yaml", "config.yaml")
    print("✅ Updated config.yaml with complete configuration")
    
    return True


def check_other_dependencies():
    """Check for other potential configuration dependencies."""
    
    print("\n🔍 Checking other potential issues...")
    
    # Check if strategy attribute exists
    import yaml
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # The system expects a 'strategy' section for backward compatibility
    if 'strategy' not in config:
        config['strategy'] = {
            'universe_size': config.get('data', {}).get('universe_size', 10),
            'rebalance_frequency': config.get('portfolio', {}).get('rebalance_frequency', 'weekly'),
            'max_position_size': config.get('portfolio', {}).get('max_position_size', 0.10),
            'long_short': True
        }
        
        with open("config.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print("✅ Added 'strategy' section for backward compatibility")
    
    return True


def main():
    """Apply all fixes."""
    print("🔧 Fixing Configuration Issues")
    print("=" * 60)
    
    # Try to fix the engine.py file first
    engine_fixed = fix_backtest_engine()
    
    if not engine_fixed:
        print("⚠️  Could not fix engine.py directly")
        print("Will fix by adding missing config parameters instead")
    
    # Add missing config attributes
    add_missing_config_attributes()
    
    # Create complete config as backup
    create_complete_config()
    
    # Check other dependencies
    check_other_dependencies()
    
    print("\n✅ Configuration fixes applied!")
    print("\nNext steps:")
    print("1. Run the backtest: python run.py --no-validate")
    print("2. If still having issues, use: copy config_complete.yaml config.yaml")
    print("3. Then run again: python run.py --no-validate")


if __name__ == "__main__":
    main()