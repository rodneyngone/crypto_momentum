#!/usr/bin/env python3
"""Test signal generation."""

from pathlib import Path
import pandas as pd
from crypto_momentum_backtest.signals.signal_generator import SignalGenerator
from crypto_momentum_backtest.data.json_storage import JsonStorage

# Load some test data
storage = JsonStorage(Path('data'))
df = storage.load_range(
    symbol='BTCUSDT',
    start_date=pd.Timestamp('2022-01-01'),
    end_date=pd.Timestamp('2022-01-31')
)

if not df.empty:
    print(f"Loaded {len(df)} days of data")
    
    # Create signal generator
    generator = SignalGenerator()
    
    # Test signal generation
    signals = generator.generate_signals(df, signal_type='momentum', symbol='BTCUSDT')
    
    print(f"Generated signals: {type(signals)}")
    print(f"Signal values: {signals.value_counts()}")
    print(f"Total signals: {(signals != 0).sum()}")
else:
    print("No data found")
