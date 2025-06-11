#!/usr/bin/env python3
"""
Fix date ranges in test scripts and config to match available data.
"""

from pathlib import Path
import yaml
import pandas as pd


def check_actual_date_range():
    """Check the actual date range of available data."""
    
    print("Checking actual date ranges in data...")
    
    from crypto_momentum_backtest.data.json_storage import JsonStorage
    
    storage = JsonStorage(Path('data'))
    
    # Check BTCUSDT date range
    # Try different date ranges to find what works
    test_ranges = [
        ('2022-09-01', '2023-12-31'),
        ('2022-09-15', '2023-12-31'),
        ('2022-10-01', '2023-12-31'),
        ('2021-12-01', '2023-12-31'),
    ]
    
    for start, end in test_ranges:
        df = storage.load_range(
            symbol='BTCUSDT',
            start_date=pd.Timestamp(start),
            end_date=pd.Timestamp(end)
        )
        
        if not df.empty:
            print(f"\n[OK] Found data from {start} to {end}")
            print(f"Actual data range: {df.index[0]} to {df.index[-1]}")
            print(f"Total days: {len(df)}")
            return df.index[0], df.index[-1]
    
    print("[ERROR] Could not find data in any test range")
    return None, None


def update_config_dates(start_date, end_date):
    """Update config.yaml with correct date range."""
    
    config_file = Path("config.yaml")
    
    if not config_file.exists():
        print("[ERROR] config.yaml not found")
        return False
    
    # Load config
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Update dates
    if 'data' in config:
        config['data']['start_date'] = start_date.strftime('%Y-%m-%d')
        config['data']['end_date'] = end_date.strftime('%Y-%m-%d')
        
        # Save updated config
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"[OK] Updated config.yaml with dates: {start_date.date()} to {end_date.date()}")
        return True
    
    return False


def create_fixed_test_script(start_date, end_date):
    """Create test script with correct date range."""
    
    test_code = f'''#!/usr/bin/env python3
"""Test signal generation with correct date range."""

from pathlib import Path
import pandas as pd
from crypto_momentum_backtest.signals.signal_generator import SignalGenerator
from crypto_momentum_backtest.data.json_storage import JsonStorage

print("Testing signal generation...")

# Load data with correct date range
storage = JsonStorage(Path('data'))
df = storage.load_range(
    symbol='BTCUSDT',
    start_date=pd.Timestamp('{start_date.strftime("%Y-%m-%d")}'),
    end_date=pd.Timestamp('{end_date.strftime("%Y-%m-%d")}')
)

if not df.empty:
    print(f"[OK] Loaded {{len(df)}} days of data")
    print(f"Date range: {{df.index[0]}} to {{df.index[-1]}}")
    
    # Create signal generator with crypto-optimized parameters
    generator = SignalGenerator(
        momentum_threshold=0.01,
        mean_return_ewm_threshold=0.003,
        mean_return_simple_threshold=0.005
    )
    
    # Test different signal types
    signal_types = ['momentum', 'mean_return_ewm', 'mean_return_simple']
    
    for signal_type in signal_types:
        print(f"\\nTesting {{signal_type}} signals...")
        
        try:
            signals = generator.generate_signals(
                df, 
                signal_type=signal_type, 
                symbol='BTCUSDT'
            )
            
            if isinstance(signals, pd.Series):
                long_signals = (signals == 1).sum()
                short_signals = (signals == -1).sum()
                total_signals = long_signals + short_signals
                
                print(f"  Long signals: {{long_signals}}")
                print(f"  Short signals: {{short_signals}}")
                print(f"  Total signals: {{total_signals}}")
                print(f"  Signal frequency: {{total_signals / len(df) * 100:.1f}}%")
            else:
                print(f"  Unexpected return type: {{type(signals)}}")
                
        except Exception as e:
            print(f"  Error: {{e}}")
    
    print("\\n[OK] Signal generation test complete!")
else:
    print("[ERROR] No data found for the specified date range")
'''
    
    with open('test_signal_fixed.py', 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    print("[OK] Created test_signal_fixed.py with correct dates")


def create_run_command(start_date, end_date):
    """Create run command with correct dates."""
    
    command = f'''# Run backtest with correct date range:
python run.py --start-date {start_date.strftime("%Y-%m-%d")} --end-date {end_date.strftime("%Y-%m-%d")} --no-validate

# Or just use:
python run.py --no-validate
# (if config.yaml has been updated with correct dates)
'''
    
    with open('run_command.txt', 'w', encoding='utf-8') as f:
        f.write(command)
    
    print("\n[OK] Created run_command.txt with correct command")
    print(command)


def main():
    print("Fixing Date Ranges to Match Available Data")
    print("=" * 60)
    
    # Check actual date range
    start_date, end_date = check_actual_date_range()
    
    if start_date is None:
        print("\n[ERROR] Could not determine data date range")
        return
    
    # Update config
    print("\nUpdating configuration...")
    update_config_dates(start_date, end_date)
    
    # Create fixed test script
    print("\nCreating fixed test script...")
    create_fixed_test_script(start_date, end_date)
    
    # Create run command
    create_run_command(start_date, end_date)
    
    print("\n[OK] All fixes applied!")
    print("\nNext steps:")
    print("1. Test signals: python test_signal_fixed.py")
    print("2. Run backtest: python run.py --no-validate")


if __name__ == "__main__":
    main()