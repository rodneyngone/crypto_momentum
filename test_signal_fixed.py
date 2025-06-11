#!/usr/bin/env python3
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
    start_date=pd.Timestamp('2022-09-15'),
    end_date=pd.Timestamp('2023-12-31')
)

if not df.empty:
    print(f"[OK] Loaded {len(df)} days of data")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Create signal generator with crypto-optimized parameters
    generator = SignalGenerator(
        momentum_threshold=0.01,
        mean_return_ewm_threshold=0.003,
        mean_return_simple_threshold=0.005
    )
    
    # Test different signal types
    signal_types = ['momentum', 'mean_return_ewm', 'mean_return_simple']
    
    for signal_type in signal_types:
        print(f"\nTesting {signal_type} signals...")
        
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
                
                print(f"  Long signals: {long_signals}")
                print(f"  Short signals: {short_signals}")
                print(f"  Total signals: {total_signals}")
                print(f"  Signal frequency: {total_signals / len(df) * 100:.1f}%")
            else:
                print(f"  Unexpected return type: {type(signals)}")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n[OK] Signal generation test complete!")
else:
    print("[ERROR] No data found for the specified date range")
