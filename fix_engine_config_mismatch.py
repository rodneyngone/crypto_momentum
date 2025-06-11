#!/usr/bin/env python3
"""
Fix the configuration mismatch between BacktestEngine and Config classes.
"""

from pathlib import Path
import re


def fix_backtest_engine_init():
    """Fix BacktestEngine to get ATR parameters from the correct location."""
    
    engine_file = Path("crypto_momentum_backtest/backtest/engine.py")
    
    if not engine_file.exists():
        print(f"❌ Could not find {engine_file}")
        return False
    
    # Read the file
    with open(engine_file, 'r') as f:
        content = f.read()
    
    # Find the PositionSizer initialization section
    # The atr_period and atr_multiplier should come from signals config or use defaults
    
    # Replace the PositionSizer initialization
    old_pattern = r'self\.position_sizer = PositionSizer\(\s*atr_period=config\.risk\.atr_period,\s*atr_multiplier=config\.risk\.atr_multiplier,'
    
    new_init = '''self.position_sizer = PositionSizer(
            atr_period=config.signals.adx_period,  # Use ADX period as ATR period
            atr_multiplier=2.0,  # Default ATR multiplier for crypto
            max_position_size=config.strategy.max_position_size,
            logger=self.logger
        )'''
    
    # Try regex replacement first
    if re.search(old_pattern, content):
        content = re.sub(old_pattern, new_init.replace('(', r'\(').replace(')', r'\)'), content)
        print("✅ Fixed PositionSizer initialization with regex")
    else:
        # Try simpler replacement
        # Look for the PositionSizer initialization block
        if "self.position_sizer = PositionSizer(" in content:
            # Find the complete initialization block
            start_idx = content.find("self.position_sizer = PositionSizer(")
            
            # Find the matching closing parenthesis
            paren_count = 0
            end_idx = start_idx
            found_start = False
            
            for i in range(start_idx, len(content)):
                if content[i] == '(':
                    if not found_start:
                        found_start = True
                    paren_count += 1
                elif content[i] == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx > start_idx:
                # Replace the entire initialization
                old_init = content[start_idx:end_idx]
                content = content.replace(old_init, new_init)
                print("✅ Fixed PositionSizer initialization")
            else:
                print("❌ Could not find PositionSizer initialization block")
                return False
        else:
            print("⚠️  PositionSizer initialization not found in expected format")
            return False
    
    # Write back the fixed content
    with open(engine_file, 'w') as f:
        f.write(content)
    
    print("✅ Updated engine.py")
    return True


def create_minimal_working_config():
    """Create a minimal working configuration."""
    
    config_content = """# Minimal working configuration

data:
  start_date: '2022-01-01'
  end_date: '2023-12-31'
  universe_size: 10

signals:
  signal_strategy: momentum
  momentum_threshold: 0.01
  momentum_ewma_span: 20
  adx_period: 14
  adx_threshold: 25.0
  ewma_fast: 10
  ewma_slow: 30
  volume_filter_multiple: 1.5
  use_volume_confirmation: true
  volume_threshold: 1.5

portfolio:
  initial_capital: 1000000.0
  max_position_size: 0.10
  min_position_size: 0.02
  rebalance_frequency: weekly
  optimization_method: risk_parity

risk:
  max_drawdown: 0.25
  max_correlation: 0.80
  max_exchange_exposure: 0.40
  volatility_regime_multiplier: 1.5
  max_drawdown_threshold: 0.25

costs:
  maker_fee: 0.001
  taker_fee: 0.001
  base_spread: 0.0005
  funding_lookback_days: 30

strategy:
  universe_size: 10
  rebalance_frequency: weekly
  max_position_size: 0.10
  long_short: true

backtest:
  benchmark_symbol: BTCUSDT
  output_directory: output
"""
    
    # Save as backup
    with open("config_minimal.yaml", 'w') as f:
        f.write(config_content)
    
    print("✅ Created config_minimal.yaml")
    
    # Also clean the current config.yaml
    clean_current_config()
    
    return True


def clean_current_config():
    """Remove atr_period and atr_multiplier from risk section of current config."""
    
    import yaml
    
    config_file = Path("config.yaml")
    
    if not config_file.exists():
        print("❌ config.yaml not found")
        return False
    
    try:
        # Load config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Remove problematic fields from risk section
        if 'risk' in config:
            config['risk'].pop('atr_period', None)
            config['risk'].pop('atr_multiplier', None)
            print("✅ Removed atr_period and atr_multiplier from risk config")
        
        # Ensure strategy section exists
        if 'strategy' not in config:
            config['strategy'] = {
                'universe_size': config.get('data', {}).get('universe_size', 10),
                'rebalance_frequency': config.get('portfolio', {}).get('rebalance_frequency', 'weekly'),
                'max_position_size': config.get('portfolio', {}).get('max_position_size', 0.10),
                'long_short': True
            }
            print("✅ Added strategy section")
        
        # Save cleaned config
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print("✅ Cleaned config.yaml")
        return True
        
    except Exception as e:
        print(f"❌ Error cleaning config: {e}")
        return False


def show_expected_structure():
    """Show what the BacktestEngine expects vs what Config provides."""
    
    print("\n📋 Configuration Structure Analysis:")
    print("=" * 60)
    
    print("\nBacktestEngine expects:")
    print("  - config.signals.adx_period (for ATR period)")
    print("  - config.strategy.max_position_size")
    print("  - config.strategy.long_short")
    print("  - config.risk.max_drawdown_threshold")
    print("  - config.risk.max_correlation")
    print("  - config.risk.max_exchange_exposure")
    print("  - config.risk.volatility_regime_multiplier")
    print("  - config.costs.maker_fee")
    print("  - config.costs.taker_fee")
    print("  - config.costs.base_spread")
    print("  - config.costs.funding_lookback_days")
    
    print("\nRiskConfig dataclass accepts:")
    print("  - max_drawdown")
    print("  - daily_var_limit") 
    print("  - position_var_limit")
    print("  - correlation_threshold")
    print("  - volatility_lookback")
    print("  - volatility_multiple")
    print("  - use_stop_losses")
    print("  - stop_loss_pct")
    print("  - (but NOT atr_period or atr_multiplier)")


def main():
    """Apply all fixes."""
    print("🔧 Fixing Configuration Mismatch")
    print("=" * 60)
    
    # First, clean the current config
    print("\n1️⃣ Cleaning current configuration...")
    clean_current_config()
    
    # Fix the BacktestEngine initialization
    print("\n2️⃣ Fixing BacktestEngine initialization...")
    engine_fixed = fix_backtest_engine_init()
    
    if not engine_fixed:
        print("⚠️  Could not automatically fix engine.py")
        print("You may need to manually edit the file")
    
    # Create minimal config as backup
    print("\n3️⃣ Creating minimal working configuration...")
    create_minimal_working_config()
    
    # Show structure analysis
    show_expected_structure()
    
    print("\n✅ Fixes applied!")
    print("\n📋 Next steps:")
    print("1. Try running again: python run.py --no-validate")
    print("2. If still having issues, use minimal config:")
    print("   copy config_minimal.yaml config.yaml")
    print("   python run.py --no-validate")
    
    print("\n💡 The issue was:")
    print("- RiskConfig doesn't accept atr_period/atr_multiplier")
    print("- These should be passed to PositionSizer directly")
    print("- BacktestEngine needs to be updated to not expect these in risk config")


if __name__ == "__main__":
    main()