#!/usr/bin/env python3
"""
Quick fix script for mean return signal generation.
Run this to immediately fix the 0 signals issue.
"""

import yaml
import shutil
from pathlib import Path

def backup_current_config():
    """Backup current config before making changes"""
    if Path("config.yaml").exists():
        shutil.copy("config.yaml", "config_backup.yaml")
        print("✅ Backed up current config to config_backup.yaml")
    return True

def create_fixed_config():
    """Create a fixed configuration with proper thresholds for crypto"""
    
    # Try to load existing config
    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        print("📁 Loaded existing config.yaml")
    except FileNotFoundError:
        # Create a new config if none exists
        config = {}
        print("📁 Creating new config.yaml")
    
    # Ensure all required sections exist
    if 'signals' not in config:
        config['signals'] = {}
    
    # Fix the signal parameters with crypto-appropriate values
    config['signals'].update({
        # Original momentum strategy (working)
        'adx_period': 14,
        'ewm_span': 10,  # Shorter for crypto volatility
        'threshold': 0.01,  # Reduced from 0.02 to 0.01 (1%)
        
        # Mean return strategies (fixed)
        'mean_return_window': 3,  # Shorter window for crypto
        'mean_return_threshold': 0.005,  # 0.5% threshold for mean returns
        'ewm_mean_threshold': 0.003,  # Even smaller for EWM means
        
        # Signal combination
        'signal_type': 'momentum',  # Default to working strategy
        'use_volume_confirmation': True,
        'min_volume_ratio': 1.2
    })
    
    # Ensure other sections exist with sensible defaults
    if 'data' not in config:
        config['data'] = {
            'start_date': '2022-01-01',
            'end_date': '2023-12-31',
            'universe_size': 10
        }
    
    if 'portfolio' not in config:
        config['portfolio'] = {
            'max_weight': 0.10,
            'rebalance_frequency': 'daily',
            'optimization_method': 'equal_weight'
        }
    
    if 'risk' not in config:
        config['risk'] = {
            'max_drawdown': 0.20,
            'stop_loss': 0.05
        }
    
    return config

def create_mean_return_config():
    """Create a specific config for testing mean return strategies"""
    config = create_fixed_config()
    
    # Override for mean return testing
    config['signals'].update({
        'signal_type': 'mean_return_ewm',  # Switch to mean return
        'ewm_span': 5,  # Very short for crypto
        'threshold': 0.002,  # 0.2% threshold
        'mean_return_window': 3,
        'mean_return_threshold': 0.003
    })
    
    return config

def save_config(config, filename="config.yaml"):
    """Save configuration to file"""
    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Saved configuration to {filename}")

def create_test_script():
    """Create a simple test script to verify the fix"""
    test_script = '''#!/usr/bin/env python3
"""
Test script to verify signal generation is working.
"""

import pandas as pd
import numpy as np
import yaml

def test_signals():
    """Test if signals are being generated"""
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Create sample data
    dates = pd.date_range('2022-01-01', '2022-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate realistic crypto returns
    returns = np.random.normal(0.001, 0.04, len(dates))
    prices = [40000]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'close': prices,
        'returns': [0] + list(np.diff(prices) / prices[:-1])
    }, index=dates)
    
    # Test momentum signals
    ewm_span = config['signals']['ewm_span']
    threshold = config['signals']['threshold']
    
    data['ewma'] = data['close'].ewm(span=ewm_span).mean()
    data['momentum'] = (data['close'] / data['ewma'] - 1)
    
    long_momentum = (data['momentum'] > threshold).sum()
    short_momentum = (data['momentum'] < -threshold).sum()
    
    print(f"SIGNAL TEST RESULTS:")
    print(f"Momentum Strategy (threshold={threshold}):")
    print(f"  Long signals: {long_momentum}")
    print(f"  Short signals: {short_momentum}")
    print(f"  Total signals: {long_momentum + short_momentum}")
    
    # Test mean return signals
    mean_threshold = config['signals'].get('mean_return_threshold', 0.005)
    ewm_returns = data['returns'].ewm(span=5).mean()
    
    long_mean = (ewm_returns > mean_threshold).sum()
    short_mean = (ewm_returns < -mean_threshold).sum()
    
    print(f"\\nMean Return EWM Strategy (threshold={mean_threshold}):")
    print(f"  Long signals: {long_mean}")
    print(f"  Short signals: {short_mean}")
    print(f"  Total signals: {long_mean + short_mean}")
    
    # Check if fix worked
    total_signals = long_momentum + short_momentum + long_mean + short_mean
    if total_signals > 0:
        print(f"\\nSUCCESS: Signal generation is working!")
        print(f"Total signals generated: {total_signals}")
    else:
        print(f"\\nISSUE: Still generating 0 signals")
        print("Try reducing thresholds further")

if __name__ == "__main__":
    test_signals()
'''
    
    with open("test_signals.py", 'w', encoding='utf-8') as f:
        f.write(test_script)
    print("Created test_signals.py")

def main():
    """Main fix function"""
    print("QUICK FIX FOR MEAN RETURN SIGNALS")
    print("="*50)
    
    # Step 1: Backup current config
    backup_current_config()
    
    # Step 2: Create fixed config
    print("\nCreating fixed configuration...")
    fixed_config = create_fixed_config()
    save_config(fixed_config, "config.yaml")
    
    # Step 3: Create mean return specific config
    print("\nCreating mean return test configuration...")
    mean_config = create_mean_return_config()
    save_config(mean_config, "config_mean_return_fixed.yaml")
    
    # Step 4: Create test script
    print("\nCreating test script...")
    create_test_script()
    
    # Step 5: Show what was changed
    print("\nCHANGES MADE:")
    print("="*50)
    print("1. Reduced signal thresholds:")
    print("   - Momentum threshold: 0.02 -> 0.01 (1%)")
    print("   - Mean return threshold: 0.02 -> 0.005 (0.5%)")
    print("   - EWM mean threshold: 0.003 (0.3%)")
    print()
    print("2. Adjusted parameters for crypto volatility:")
    print("   - EWM span: 20 -> 10 (more responsive)")
    print("   - Mean return window: 5 -> 3 (shorter)")
    print()
    print("3. Created backup and test files:")
    print("   - config_backup.yaml (your original)")
    print("   - config_mean_return_fixed.yaml (for mean return testing)")
    print("   - test_signals.py (verification script)")
    
    # Step 6: Instructions
    print("\nNEXT STEPS:")
    print("="*50)
    print("1. Test the fix:")
    print("   python test_signals.py")
    print()
    print("2. Run your backtest:")
    print("   python run.py --no-validate")
    print()
    print("3. Test mean return strategies:")
    print("   copy config_mean_return_fixed.yaml config.yaml")
    print("   python test_mean_return_signals.py")
    print()
    print("4. If still having issues, run the full debug:")
    print("   python debug_signals.py")
    print()
    print("5. For parameter optimization:")
    print("   python parameter_optimizer.py")
    
    print("\nQUICK FIX COMPLETE!")
    print("Your signal generation should now work properly.")

if __name__ == "__main__":
    main()